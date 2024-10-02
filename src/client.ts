const ws = new WebSocket('ws://localhost:8000/showdown/websocket');

ws.onopen = () => {
  console.log('WebSocket connection opened');
  ws.send('|/trn basketbasket|')
  ws.send('|/search gen1randombattle|');
};

ws.onmessage = (event) => {
  console.log('Received message:', event.data);
};

ws.onclose = (event) => {
  console.log('WebSocket closed:', event);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
