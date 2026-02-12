class AIAvatarClient {
    constructor() {
        this.pc = null;
        this.pcId = null;
        this.localStream = null;
        this.remoteAudioTrack = null;
        this.remoteVideoTrack = null;
        this.isConnected = false;
        this.dataChannel = null;
        this.audioContext = null;
        this.audioAnalyser = null;
        this.audioLevelInterval = null;
        
        // Conversation state
        this.currentState = 'idle'; // idle, listening, thinking, speaking
        this.hasMessages = false;
        
        this.elements = {
            video: document.getElementById('avatar-video'),
            image: document.getElementById('avatar-image'),
            placeholder: document.getElementById('avatar-placeholder'),
            avatarUpload: document.getElementById('avatar-upload'),
            voiceSelect: document.getElementById('voice-select'),
            connectBtn: document.getElementById('connect-btn'),
            disconnectBtn: document.getElementById('disconnect-btn'),
            status: document.getElementById('connection-status'),
            textInput: document.getElementById('text-input'),
            sendBtn: document.getElementById('send-btn'),
            conversation: document.getElementById('conversation'),
            transcript: document.getElementById('transcript'),
            // New elements
            stateOverlay: document.getElementById('avatar-state-overlay'),
            stateListening: document.getElementById('state-listening'),
            stateThinking: document.getElementById('state-thinking'),
            stateSpeaking: document.getElementById('state-speaking'),
            audioMeterBar: document.getElementById('audio-meter-bar'),
            audioLevelText: document.getElementById('audio-level-text'),
            liveTranscription: document.getElementById('live-transcription'),
            liveText: document.getElementById('live-text'),
        };
        
        this.bindEvents();
        this.loadVoices();
    }
    
    bindEvents() {
        this.elements.avatarUpload.addEventListener('change', (e) => this.uploadAvatar(e));
        this.elements.voiceSelect.addEventListener('change', (e) => this.setVoice(e.target.value));
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
        
        // Text input handlers
        this.elements.sendBtn.addEventListener('click', () => this.sendTextMessage());
        this.elements.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendTextMessage();
            }
        });
    }
    
    async loadVoices() {
        try {
            const response = await fetch('/api/voices');
            const data = await response.json();
            this.elements.voiceSelect.value = data.current;
        } catch (e) {
            console.error('Failed to load voices:', e);
        }
    }
    
    async uploadAvatar(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/avatar', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    this.elements.image.src = e.target.result;
                    this.elements.image.classList.add('visible');
                    this.elements.placeholder.classList.add('hidden');
                    this.elements.video.style.display = 'none';
                };
                reader.readAsDataURL(file);
                
                this.log('Avatar uploaded successfully');
                this.addToConversation('system', 'Avatar uploaded! Click Connect to start chatting.');
            }
        } catch (e) {
            console.error('Failed to upload avatar:', e);
            this.addToConversation('system', 'Failed to upload avatar');
        }
    }
    
    async setVoice(voice) {
        try {
            console.log(`Setting voice to: ${voice}`);
            const formData = new FormData();
            formData.append('voice', voice);
            
            const response = await fetch('/api/voice', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            console.log('Voice change response:', result);
            
            this.log(`Voice set to: ${voice}`);
        } catch (e) {
            console.error('Failed to set voice:', e);
        }
    }
    
    async sendTextMessage() {
        const text = this.elements.textInput.value.trim();
        if (!text || !this.isConnected) return;
        
        this.elements.textInput.value = '';
        this.addToConversation('user', text);
        this.setState('thinking');
        
        try {
            // Send text message to server
            const response = await fetch('/api/text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: text,
                    pc_id: this.pcId 
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to send message');
            }
            
            this.log(`Sent text: ${text}`);
        } catch (e) {
            console.error('Failed to send text:', e);
            this.addToConversation('system', 'Error sending message');
            this.setState('idle');
        }
    }
    
    setState(state) {
        this.currentState = state;
        
        // Hide all state indicators
        this.elements.stateListening.classList.remove('active');
        this.elements.stateThinking.classList.remove('active');
        this.elements.stateSpeaking.classList.remove('active');
        
        // Show overlay only when connected
        if (this.isConnected && state !== 'idle') {
            this.elements.stateOverlay.classList.remove('hidden');
            
            switch (state) {
                case 'listening':
                    this.elements.stateListening.classList.add('active');
                    break;
                case 'thinking':
                    this.elements.stateThinking.classList.add('active');
                    break;
                case 'speaking':
                    this.elements.stateSpeaking.classList.add('active');
                    break;
            }
        } else {
            this.elements.stateOverlay.classList.add('hidden');
        }
    }
    
    addToConversation(role, text) {
        // Remove empty state if this is first real message
        if (!this.hasMessages) {
            const emptyState = this.elements.conversation.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }
            this.hasMessages = true;
        }
        
        const div = document.createElement('div');
        div.className = `message ${role}`;
        
        if (role !== 'system') {
            const roleLabel = document.createElement('div');
            roleLabel.className = 'role';
            roleLabel.textContent = role === 'user' ? 'You' : 'AI';
            div.appendChild(roleLabel);
        }
        
        const content = document.createElement('div');
        content.className = 'content';
        content.textContent = text;
        div.appendChild(content);
        
        this.elements.conversation.appendChild(div);
        this.elements.conversation.scrollTop = this.elements.conversation.scrollHeight;
    }
    
    showLiveTranscription(text) {
        if (text) {
            this.elements.liveText.textContent = text;
            this.elements.liveTranscription.classList.remove('hidden');
        } else {
            this.elements.liveTranscription.classList.add('hidden');
            this.elements.liveText.textContent = '';
        }
    }
    
    setupAudioMeter(stream) {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.audioAnalyser = this.audioContext.createAnalyser();
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.audioAnalyser);
            this.audioAnalyser.fftSize = 256;
            
            const dataArray = new Uint8Array(this.audioAnalyser.frequencyBinCount);
            
            const updateLevel = () => {
                if (!this.isConnected) return;
                
                this.audioAnalyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
                const level = Math.min(100, Math.round((average / 128) * 100));
                
                this.elements.audioMeterBar.style.width = `${level}%`;
                this.elements.audioLevelText.textContent = `${level}%`;
            };
            
            this.audioLevelInterval = setInterval(updateLevel, 50);
        } catch (e) {
            console.error('Failed to setup audio meter:', e);
        }
    }
    
    async connect() {
        try {
            this.setStatus('connecting', '🔄 Connecting...');
            this.log('Starting WebRTC connection...');

            // Fetch ICE configuration from server (includes TURN relay for WSL)
            let iceConfig = {
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            };
            try {
                const iceResp = await fetch('/api/ice-config');
                if (iceResp.ok) {
                    iceConfig = await iceResp.json();
                    this.log(`ICE config: ${iceConfig.iceTransportPolicy || 'all'} mode, ${iceConfig.iceServers?.length || 0} servers`);
                    if (iceConfig.iceTransportPolicy === 'relay') {
                        this.log('Using TURN relay (WSL2 mode)');
                    }
                }
            } catch (e) {
                this.log('Could not fetch ICE config, using defaults');
            }
            
            // Get microphone access with noise suppression
            this.log('Requesting microphone access...');
            this.localStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                },
                video: false
            });
            this.log('Microphone access granted');
            
            // Setup audio level meter
            this.setupAudioMeter(this.localStream);
            
            // Create peer connection with server-provided ICE config
            this.pc = new RTCPeerConnection(iceConfig);
            
            // Add local audio track - this creates the sender for our mic
            const audioTrack = this.localStream.getAudioTracks()[0];
            if (audioTrack) {
                this.log(`Adding local audio track: ${audioTrack.label}, enabled: ${audioTrack.enabled}, muted: ${audioTrack.muted}`);
                // Use addTransceiver for bidirectional audio (send mic, receive TTS)
                const audioTransceiver = this.pc.addTransceiver(audioTrack, { 
                    direction: 'sendrecv',
                    streams: [this.localStream]
                });
                this.log(`Audio transceiver created, direction: ${audioTransceiver.direction}`);
            } else {
                this.log('ERROR: No audio track from microphone!');
            }
            
            // Add transceiver for receiving video only
            this.pc.addTransceiver('video', { direction: 'recvonly' });
            
            // Create data channel for receiving transcriptions and events
            this.dataChannel = this.pc.createDataChannel('events');
            this.dataChannel.onmessage = (event) => {
                this.handleDataChannelMessage(event.data);
            };
            this.dataChannel.onopen = () => {
                this.log('Data channel opened');
            };
            
            // Handle incoming tracks
            this.pc.ontrack = (event) => {
                this.log(`Received ${event.track.kind} track: ${event.track.id}`);
                console.log('Track event:', event);
                
                if (event.track.kind === 'audio') {
                    this.remoteAudioTrack = event.track;
                    const audio = new Audio();
                    audio.srcObject = new MediaStream([event.track]);
                    audio.play().catch(e => console.error('Audio play error:', e));
                } else if (event.track.kind === 'video') {
                    this.remoteVideoTrack = event.track;
                    this.elements.video.srcObject = new MediaStream([event.track]);
                    this.elements.video.style.display = 'block';
                    this.elements.image.classList.remove('visible');
                    this.elements.placeholder.classList.add('hidden');
                }
            };
            
            // ICE candidate handling
            this.pc.onicecandidate = (event) => {
                if (event.candidate) {
                    this.log(`ICE candidate: ${event.candidate.candidate.substring(0, 50)}...`);
                } else {
                    this.log('ICE gathering complete');
                }
            };
            
            // ICE connection state
            this.pc.oniceconnectionstatechange = () => {
                this.log(`ICE state: ${this.pc.iceConnectionState}`);
                console.log('ICE connection state:', this.pc.iceConnectionState);
                
                if (this.pc.iceConnectionState === 'connected' || 
                    this.pc.iceConnectionState === 'completed') {
                    this.onConnected();
                } else if (this.pc.iceConnectionState === 'failed') {
                    this.log('ICE connection failed');
                    this.setStatus('error', '❌ Connection failed');
                    this.disconnect();
                } else if (this.pc.iceConnectionState === 'disconnected') {
                    this.log('ICE connection disconnected');
                }
            };
            
            // Connection state
            this.pc.onconnectionstatechange = () => {
                this.log(`Connection state: ${this.pc.connectionState}`);
                console.log('Connection state:', this.pc.connectionState);
            };
            
            // Signaling state
            this.pc.onsignalingstatechange = () => {
                this.log(`Signaling state: ${this.pc.signalingState}`);
            };
            
            // Create offer
            this.log('Creating offer...');
            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);
            this.log('Local description set');
            
            // Wait for ICE gathering to complete
            this.log('Waiting for ICE gathering...');
            await this.waitForIceGathering();
            this.log('ICE gathering complete, sending offer to server...');
            
            // Send offer to server
            const response = await fetch('/api/offer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sdp: this.pc.localDescription.sdp,
                    type: this.pc.localDescription.type
                })
            });
            
            const data = await response.json();
            console.log('Server response:', data);
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            if (!data.sdp) {
                throw new Error('No SDP in server response');
            }
            
            this.pcId = data.pc_id;
            this.log(`Received answer from server (pc_id: ${this.pcId})`);
            
            // Set remote description (answer from server)
            this.log('Setting remote description...');
            await this.pc.setRemoteDescription(new RTCSessionDescription({
                type: data.type || 'answer',
                sdp: data.sdp
            }));
            this.log('Remote description set');
            
        } catch (e) {
            console.error('Failed to connect:', e);
            this.log(`Error: ${e.message}`);
            this.setStatus('error', '❌ Error: ' + e.message);
            this.disconnect();
        }
    }
    
    handleDataChannelMessage(data) {
        try {
            const msg = JSON.parse(data);
            this.log(`Data channel: ${msg.type}`);
            
            switch (msg.type) {
                case 'transcription':
                    // Final transcription - add to conversation
                    this.addToConversation('user', msg.text);
                    this.showLiveTranscription(null);
                    this.setState('thinking');
                    break;
                    
                case 'interim_transcription':
                    // Live transcription as user speaks
                    this.showLiveTranscription(msg.text);
                    break;
                    
                case 'response':
                    // AI response text
                    this.addToConversation('assistant', msg.text);
                    break;
                    
                case 'user_started_speaking':
                    this.setState('listening');
                    this.showLiveTranscription('Listening...');
                    break;
                    
                case 'user_stopped_speaking':
                    this.showLiveTranscription('Processing...');
                    this.setState('thinking');
                    break;
                    
                case 'bot_started_speaking':
                    this.setState('speaking');
                    this.showLiveTranscription(null);
                    break;
                    
                case 'bot_stopped_speaking':
                    this.setState('idle');
                    break;
                    
                case 'llm_started':
                    this.setState('thinking');
                    break;
            }
        } catch (e) {
            this.log(`Data channel message (raw): ${data}`);
        }
    }
    
    async waitForIceGathering() {
        if (this.pc.iceGatheringState === 'complete') {
            return;
        }
        
        return new Promise((resolve) => {
            const checkState = () => {
                if (this.pc.iceGatheringState === 'complete') {
                    resolve();
                }
            };
            
            this.pc.onicegatheringstatechange = checkState;
            
            // Timeout after 5 seconds
            setTimeout(() => {
                this.log('ICE gathering timeout, proceeding anyway');
                resolve();
            }, 5000);
        });
    }
    
    onConnected() {
        this.isConnected = true;
        this.setStatus('connected', '🟢 Connected');
        this.elements.connectBtn.disabled = true;
        this.elements.disconnectBtn.disabled = false;
        this.elements.textInput.disabled = false;
        this.elements.sendBtn.disabled = false;
        this.elements.stateOverlay.classList.remove('hidden');
        this.log('WebRTC connection established!');
        this.addToConversation('system', 'Connected! Start speaking or type a message.');
    }
    
    disconnect() {
        this.log('Disconnecting...');
        
        // Stop audio level monitoring
        if (this.audioLevelInterval) {
            clearInterval(this.audioLevelInterval);
            this.audioLevelInterval = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }
        
        this.elements.video.srcObject = null;
        this.elements.video.style.display = 'none';
        
        // Show avatar image if available
        if (this.elements.image.src && this.elements.image.src !== window.location.href) {
            this.elements.image.classList.add('visible');
        } else {
            this.elements.placeholder.classList.remove('hidden');
        }
        
        // Reset audio meter
        this.elements.audioMeterBar.style.width = '0%';
        this.elements.audioLevelText.textContent = '0%';
        
        this.isConnected = false;
        this.pcId = null;
        this.dataChannel = null;
        this.elements.connectBtn.disabled = false;
        this.elements.disconnectBtn.disabled = true;
        this.elements.textInput.disabled = true;
        this.elements.sendBtn.disabled = true;
        this.elements.stateOverlay.classList.add('hidden');
        this.elements.liveTranscription.classList.add('hidden');
        this.setState('idle');
        this.setStatus('disconnected', '⚫ Disconnected');
    }
    
    setStatus(type, text) {
        this.elements.status.textContent = text;
        this.elements.status.className = `status-${type}`;
    }
    
    log(message) {
        console.log(`[AIAvatar] ${message}`);
        const div = document.createElement('div');
        const time = new Date().toLocaleTimeString();
        div.textContent = `[${time}] ${message}`;
        this.elements.transcript.appendChild(div);
        this.elements.transcript.scrollTop = this.elements.transcript.scrollHeight;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.avatarClient = new AIAvatarClient();
});
