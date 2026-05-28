import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { Notice } from "../components/primitives";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

// One coarse-grained boundary at the panel layer. We render the error to
// the user with a clear hint that the failure is local to the panel —
// the topbar health probe still works in the background.

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // We deliberately do not report errors externally; this is a local
    // research tool. The console is enough.
     
    console.error("[gui] panel error:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="p-4">
          <Notice tone="err" title="This panel crashed">
            <p className="mt-1 font-mono text-xs">{this.state.error.message}</p>
            <p className="mt-2 text-xs">
              Refresh the page to recover. The backend run history and artifacts
              are not affected.
            </p>
          </Notice>
        </div>
      );
    }
    return this.props.children;
  }
}
