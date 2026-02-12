<# 
.SYNOPSIS
    TalkToMe - LAN Access Setup (Run as Administrator)

.DESCRIPTION
    Sets up everything needed to access TalkToMe running in WSL2
    from other devices on your local network.

    The key insight: WSL2 defaults to NAT networking, which gives WSL a
    private 172.x IP that LAN devices cannot reach. WebRTC also needs UDP,
    which netsh portproxy cannot forward.

    The solution is WSL2 "mirrored" networking (Windows 11 22H2+), which
    gives WSL the same IP as the Windows host. Combined with SSL certs
    and a firewall rule, everything just works.

    Steps performed:
      1. Enables WSL2 mirrored networking (edits .wslconfig)
      2. Generates self-signed SSL certs with your LAN IP as SAN
      3. Copies certs into WSL project
      4. Adds Windows Firewall rules (TCP + UDP)

.EXAMPLE
    .\setup_lan.ps1
    .\setup_lan.ps1 -Teardown
#>

param(
    [switch]$Teardown,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$FwRuleNameTcp = "TalkToMe Server (TCP $Port)"
$FwRuleNameUdp = "TalkToMe WebRTC Media (UDP)"
$WslDistro     = "Ubuntu"
$CertsDir      = Join-Path $PSScriptRoot "certs"
$WslProject    = "`$HOME/TalkToMe"
$WslConfigPath = Join-Path $env:USERPROFILE ".wslconfig"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Ok   { param($msg) Write-Host "  [OK]   $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "  [..] $msg" -ForegroundColor Cyan }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-LanIP {
    $adapter = Get-NetIPConfiguration |
        Where-Object { $_.IPv4DefaultGateway -ne $null -and $_.NetAdapter.Status -eq "Up" } |
        Select-Object -First 1
    return $adapter.IPv4Address.IPAddress
}

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
if ($Teardown) {
    Write-Host "`n=== TalkToMe LAN Teardown ===" -ForegroundColor Yellow

    # Remove old NAT-mode port proxy if it exists
    netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0 2>$null
    Write-Ok "Port proxy removed (if any)"

    Remove-NetFirewallRule -DisplayName $FwRuleNameTcp -ErrorAction SilentlyContinue
    Remove-NetFirewallRule -DisplayName $FwRuleNameUdp -ErrorAction SilentlyContinue
    Write-Ok "Firewall rules removed"

    Write-Host "`n  Note: .wslconfig was NOT reverted. Edit manually if needed:" -ForegroundColor Yellow
    Write-Host "    $WslConfigPath" -ForegroundColor Yellow
    Write-Host "`nTeardown complete." -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  TalkToMe - LAN Access Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Admin)) {
    Write-Fail "This script must be run as Administrator."
    Write-Host "  Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    exit 1
}
Write-Ok "Running as Administrator"

# -------------------------------------------------------------------------
# Step 1: Enable WSL2 mirrored networking
# -------------------------------------------------------------------------
Write-Host "`nStep 1: Configuring WSL2 mirrored networking..." -ForegroundColor White

$needsRestart = $false

if (Test-Path $WslConfigPath) {
    $wslContent = Get-Content $WslConfigPath -Raw
}
else {
    $wslContent = ""
}

if ($wslContent -match 'networkingMode\s*=\s*mirrored') {
    Write-Ok "Mirrored networking already enabled in .wslconfig"
}
else {
    # Add or update the [wsl2] section
    if ($wslContent -match '\[wsl2\]') {
        # Section exists, add the setting after it
        $wslContent = $wslContent -replace '(\[wsl2\])', "`$1`r`nnetworkingMode=mirrored"
    }
    else {
        # No [wsl2] section, append one
        if ($wslContent.Length -gt 0 -and -not $wslContent.EndsWith("`n")) {
            $wslContent += "`r`n"
        }
        $wslContent += "[wsl2]`r`nnetworkingMode=mirrored`r`n"
    }
    $wslContent | Set-Content -Path $WslConfigPath -Encoding UTF8 -NoNewline
    Write-Ok "Enabled networkingMode=mirrored in $WslConfigPath"
    $needsRestart = $true
}

# Also clean up any old NAT-mode port proxy
netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0 2>$null

# -------------------------------------------------------------------------
# Step 2: Detect LAN IP
# -------------------------------------------------------------------------
Write-Host "`nStep 2: Detecting network..." -ForegroundColor White

$LanIP = Get-LanIP
if (-not $LanIP) {
    Write-Fail "Could not detect LAN IP. Are you connected to a network?"
    exit 1
}
Write-Ok "LAN IP: $LanIP"

# -------------------------------------------------------------------------
# Step 3: Generate SSL certificates
# -------------------------------------------------------------------------
Write-Host "`nStep 3: Generating SSL certificates..." -ForegroundColor White

New-Item -ItemType Directory -Force -Path $CertsDir | Out-Null

$CertPem = Join-Path $CertsDir "cert.pem"
$KeyPem  = Join-Path $CertsDir "key.pem"

$needNewCert = $true
if ((Test-Path $CertPem) -and (Test-Path $KeyPem)) {
    try {
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $existingSans = & openssl x509 -in $CertPem -noout -ext subjectAltName 2>&1
        $ErrorActionPreference = $prevEAP
        $escapedIP = [regex]::Escape($LanIP)
        if ($existingSans -match $escapedIP) {
            Write-Ok "Existing cert already covers $LanIP - keeping it"
            $needNewCert = $false
        }
        else {
            Write-Info "Existing cert does not cover $LanIP - regenerating"
        }
    }
    catch {
        Write-Info "Could not read existing cert - regenerating"
    }
}

if ($needNewCert) {
    $openssl = "openssl"
    if (-not (Get-Command openssl -ErrorAction SilentlyContinue)) {
        $gitOpenSSL = "C:\Program Files\Git\usr\bin\openssl.exe"
        if (Test-Path $gitOpenSSL) {
            $openssl = $gitOpenSSL
        }
        else {
            Write-Fail "openssl not found. Install Git for Windows or add openssl to PATH."
            exit 1
        }
    }

    $opensslConf = Join-Path $CertsDir "openssl_san.cnf"
    $confLines = @(
        "[req]"
        "default_bits       = 2048"
        "prompt             = no"
        "default_md         = sha256"
        "distinguished_name = dn"
        "x509_extensions    = v3_ca"
        "req_extensions     = v3_ca"
        ""
        "[dn]"
        "CN = TalkToMe"
        ""
        "[v3_ca]"
        "subjectAltName = @alt_names"
        "basicConstraints = critical, CA:TRUE"
        "keyUsage = critical, digitalSignature, keyEncipherment"
        ""
        "[alt_names]"
        "DNS.1 = localhost"
        "IP.1  = 127.0.0.1"
        "IP.2  = $LanIP"
    )
    $confLines -join "`r`n" | Set-Content -Path $opensslConf -Encoding ASCII -NoNewline

    Write-Info "Generating cert for: localhost, 127.0.0.1, $LanIP"

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $openssl req -x509 -newkey rsa:2048 -keyout $KeyPem -out $CertPem -days 1825 -nodes -config $opensslConf 2>&1 | Out-Null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP

    if ($exitCode -ne 0) {
        Write-Fail "Failed to generate certificate with openssl (exit code $exitCode)"
        exit 1
    }

    Remove-Item $opensslConf -ErrorAction SilentlyContinue
    Write-Ok "SSL certificate generated (valid 5 years)"
    Write-Info "SANs: localhost, 127.0.0.1, $LanIP"
}

# -------------------------------------------------------------------------
# Step 4: Copy certs to WSL
# -------------------------------------------------------------------------
Write-Host "`nStep 4: Copying certificates to WSL..." -ForegroundColor White

$WslCertsDir = "$WslProject/certs"
wsl -d $WslDistro -- bash -c "mkdir -p $WslCertsDir" 2>$null

$CertPemWsl = ($CertPem -replace '\\','/') -replace '^([A-Za-z]):','/mnt/$1'
$CertPemWsl = $CertPemWsl.Substring(0,5).ToLower() + $CertPemWsl.Substring(5)
$KeyPemWsl  = ($KeyPem  -replace '\\','/') -replace '^([A-Za-z]):','/mnt/$1'
$KeyPemWsl  = $KeyPemWsl.Substring(0,5).ToLower()  + $KeyPemWsl.Substring(5)

wsl -d $WslDistro -- bash -c "cp '$CertPemWsl' '$WslCertsDir/cert.pem'" 2>$null
wsl -d $WslDistro -- bash -c "cp '$KeyPemWsl' '$WslCertsDir/key.pem'" 2>$null
Write-Ok "Certificates copied to WSL"

# -------------------------------------------------------------------------
# Step 5: Firewall rules
# -------------------------------------------------------------------------
Write-Host "`nStep 5: Configuring Windows Firewall..." -ForegroundColor White

# TCP for HTTPS signaling
if (-not (Get-NetFirewallRule -DisplayName $FwRuleNameTcp -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $FwRuleNameTcp -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Private,Domain -Description "TalkToMe HTTPS server" | Out-Null
    Write-Ok "TCP firewall rule created (port $Port)"
}
else {
    Write-Ok "TCP firewall rule exists"
}

# UDP for WebRTC media (ephemeral range)
if (-not (Get-NetFirewallRule -DisplayName $FwRuleNameUdp -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $FwRuleNameUdp -Direction Inbound -Protocol UDP -LocalPort 49152-65535 -Action Allow -Profile Private,Domain -Description "TalkToMe WebRTC media (UDP)" | Out-Null
    Write-Ok "UDP firewall rule created (ports 49152-65535)"
}
else {
    Write-Ok "UDP firewall rule exists"
}

# Also allow WSL through Hyper-V firewall if applicable
$hns = Get-NetFirewallRule -DisplayName "*Hyper-V*WSL*" -ErrorAction SilentlyContinue
if (-not $hns) {
    Write-Info "No Hyper-V/WSL firewall rules detected (OK on most setups)"
}

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  LAN Setup Complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

if ($needsRestart) {
    Write-Host "  ** WSL RESTART REQUIRED **" -ForegroundColor Red
    Write-Host "  Mirrored networking was just enabled. Run:" -ForegroundColor Yellow
    Write-Host "    wsl --shutdown" -ForegroundColor White
    Write-Host "  Then start WSL again and launch the server." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "  Your server will be accessible at:" -ForegroundColor White
Write-Host "    Local:   https://localhost:$Port" -ForegroundColor Cyan
Write-Host "    Network: https://${LanIP}:$Port" -ForegroundColor Cyan
Write-Host ""
Write-Host "  To start the server (in WSL):" -ForegroundColor White
Write-Host "    cd ~/TalkToMe && ./run.sh" -ForegroundColor Yellow
Write-Host ""
Write-Host "  From other devices on your network:" -ForegroundColor White
Write-Host "    1. Open https://${LanIP}:$Port" -ForegroundColor Yellow
Write-Host "    2. Accept the self-signed certificate warning" -ForegroundColor Yellow
Write-Host "    3. Upload an avatar and click Connect" -ForegroundColor Yellow
Write-Host ""
Write-Host "  To undo: .\setup_lan.ps1 -Teardown" -ForegroundColor Yellow
Write-Host ""
