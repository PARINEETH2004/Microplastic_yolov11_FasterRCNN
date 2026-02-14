import { useState } from 'react';

export default function ConnectionTest() {
    const [result, setResult] = useState('');
    const [error, setError] = useState('');

    const testConnection = async () => {
        try {
            setResult('Testing connection...');
            setError('');

            // Test health endpoint
            const healthResponse = await fetch('http://localhost:5000/api/health');
            const healthData = await healthResponse.json();
            setResult(`Health Check: ${JSON.stringify(healthData, null, 2)}`);

            // Test with a small image if available
            // This would require creating a small test image

        } catch (err) {
            setError(`Connection Error: ${err.message}`);
            setResult('');
        }
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'monospace' }}>
            <h2>Connection Test</h2>
            <button onClick={testConnection} style={{ padding: '10px', fontSize: '16px' }}>
                Test Backend Connection
            </button>
            <div style={{ marginTop: '20px' }}>
                {error && <div style={{ color: 'red' }}><pre>{error}</pre></div>}
                {result && <div style={{ color: 'green' }}><pre>{result}</pre></div>}
            </div>
        </div>
    );
}