import { Navigate, Route, Routes } from "react-router-dom";

import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { ErrorBoundary } from "./ErrorBoundary";

import { PipelinePanel } from "../panels/PipelinePanel";
import { RunsPanel } from "../panels/RunsPanel";
import { RunDetailPanel } from "../panels/RunDetailPanel";
import { ConfigsPanel } from "../panels/ConfigsPanel";
import { ConfigEditorPanel } from "../panels/ConfigEditorPanel";
import { InputsPanel } from "../panels/InputsPanel";
import { ArtifactsPanel } from "../panels/ArtifactsPanel";
import { MetricsPanel } from "../panels/MetricsPanel";
import { VlmPanel } from "../panels/VlmPanel";
import { ViewerPanel } from "../panels/ViewerPanel";
import { DiffPanel } from "../panels/DiffPanel";
import { ReportPanel } from "../panels/ReportPanel";
import { ScheduleComparePage } from "../pages/ScheduleCompare";

export function App() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface-0 text-ink">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="flex-1 overflow-y-auto">
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<PipelinePanel />} />
              <Route path="/runs" element={<RunsPanel />} />
              <Route path="/runs/:runId" element={<RunDetailPanel />} />
              <Route path="/configs" element={<ConfigsPanel />} />
              <Route path="/configs/:configName" element={<ConfigEditorPanel />} />
              <Route path="/inputs" element={<InputsPanel />} />
              <Route path="/artifacts" element={<ArtifactsPanel />} />
              <Route path="/metrics" element={<MetricsPanel />} />
              <Route path="/vlm" element={<VlmPanel />} />
              <Route path="/viewer" element={<ViewerPanel />} />
              <Route path="/diff" element={<DiffPanel />} />
              <Route path="/report" element={<ReportPanel />} />
              <Route path="/schedule" element={<ScheduleComparePage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
