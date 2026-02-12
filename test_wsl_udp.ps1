# Quick UDP bidirectional test for WSL2 mirrored mode
# Tests whether WSL can reach a Windows UDP socket on various addresses

Write-Host "`n=== WSL2 UDP Connectivity Test ===" -ForegroundColor Cyan
Write-Host "Testing if WSL can send UDP to Windows on different addresses`n"

$port = 19876
$results = @{}

foreach ($bindAddr in @("0.0.0.0", "192.168.0.176", "127.0.0.1")) {
    foreach ($sendTo in @("192.168.0.176", "127.0.0.1")) {
        $label = "WSL->$sendTo (Win bound to $bindAddr)"
        Write-Host "Testing: $label ..." -NoNewline

        # Start listener
        try {
            $listener = New-Object Net.Sockets.UdpClient
            $listener.Client.SetSocketOption([Net.Sockets.SocketOptionLevel]::Socket,
                [Net.Sockets.SocketOptionName]::ReuseAddress, $true)
            $listener.Client.Bind([Net.IPEndPoint]::new([Net.IPAddress]::Parse($bindAddr), $port))
            $listener.Client.ReceiveTimeout = 3000
        } catch {
            Write-Host " BIND FAILED: $_" -ForegroundColor Red
            $results[$label] = "BIND_FAIL"
            continue
        }

        # Send from WSL
        $null = wsl bash -c "python3 -c `"import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b'test_$sendTo', ('$sendTo', $port)); s.close()`" 2>&1"
        Start-Sleep -Milliseconds 200

        # Check if received
        try {
            $ep = New-Object Net.IPEndPoint([Net.IPAddress]::Any, 0)
            $data = $listener.Receive([ref]$ep)
            $msg = [Text.Encoding]::ASCII.GetString($data)
            Write-Host " OK (got '$msg' from $($ep.Address):$($ep.Port))" -ForegroundColor Green
            $results[$label] = "OK"
        } catch {
            Write-Host " FAILED (no data)" -ForegroundColor Red
            $results[$label] = "FAIL"
        }
        $listener.Close()
        Start-Sleep -Milliseconds 100
    }
}

Write-Host "`n=== Results ===" -ForegroundColor Cyan
foreach ($kv in $results.GetEnumerator()) {
    $color = if ($kv.Value -eq "OK") { "Green" } else { "Red" }
    Write-Host "  $($kv.Key): $($kv.Value)" -ForegroundColor $color
}
Write-Host ""
