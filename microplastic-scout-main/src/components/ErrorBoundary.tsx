import React from 'react';

interface ErrorBoundaryProps {
    children: React.ReactNode;
    fallback?: React.ReactNode;
}

interface ErrorBoundaryState {
    hasError: boolean;
    error?: Error;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return this.props.fallback || (
                <div className="container mx-auto px-6 py-8 text-center">
                    <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-6 max-w-2xl mx-auto">
                        <h2 className="text-xl font-semibold text-destructive mb-2">Something went wrong</h2>
                        <p className="text-destructive mb-4">
                            An error occurred while rendering this component.
                        </p>
                        <details className="text-left text-sm text-muted-foreground bg-muted/50 p-3 rounded">
                            <summary className="cursor-pointer font-medium">Error details</summary>
                            <pre className="mt-2 whitespace-pre-wrap">{this.state.error?.toString()}</pre>
                        </details>
                        <button
                            onClick={() => window.location.reload()}
                            className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded hover:opacity-90"
                        >
                            Reload Page
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}