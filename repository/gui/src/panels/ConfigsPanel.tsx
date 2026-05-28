import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { endpoints } from "../api/endpoints";
import { queryKeys } from "../api/queryKeys";
import { Card, CardHeader, Empty, Notice, Spinner } from "../components/primitives";
import { PageHeader } from "./PageHeader";
import { formatBytes, formatUnixTimestamp } from "../lib/format";

// List every YAML config the backend can see under /configs/.
// Each row links to the schema-driven editor.

export function ConfigsPanel() {
  const q = useQuery({
    queryKey: queryKeys.configs(),
    queryFn: endpoints.listConfigs,
    staleTime: 30_000,
  });

  return (
    <div>
      <PageHeader
        title="Configs"
        subtitle="Every YAML config under the configs/ directory. Click one to see its typed parameters."
      />
      <div className="space-y-4 p-6">
        {q.isLoading && (
          <div className="flex items-center gap-2 text-ink-muted">
            <Spinner /> Loading configs…
          </div>
        )}
        {q.isError && (
          <Notice tone="err" title="Could not load configs">
            {(q.error as Error).message}
          </Notice>
        )}
        {q.data && q.data.length === 0 && (
          <Empty
            title="No configs found"
            hint="The backend reports configs/ is empty. Add a YAML file there and refresh."
          />
        )}
        {q.data && q.data.length > 0 && (
          <Card>
            <CardHeader title={`${q.data.length} config files`} />
            <div className="-mx-4 overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-surface-2 text-left text-[11px] uppercase tracking-wider text-ink-muted">
                  <tr>
                    <th className="px-4 py-2">Name</th>
                    <th className="px-4 py-2">Path</th>
                    <th className="px-4 py-2">Size</th>
                    <th className="px-4 py-2">Modified</th>
                  </tr>
                </thead>
                <tbody>
                  {q.data.map((c) => (
                    <tr
                      key={c.name}
                      className="border-t border-surface-border hover:bg-surface-2/40"
                    >
                      <td className="px-4 py-2">
                        <Link
                          to={`/configs/${encodeURIComponent(c.name)}`}
                          className="font-medium text-accent hover:underline"
                        >
                          {c.name}
                        </Link>
                      </td>
                      <td className="px-4 py-2 font-mono text-xs text-ink-muted">
                        {c.path}
                      </td>
                      <td className="px-4 py-2 text-xs">{formatBytes(c.size_bytes)}</td>
                      <td className="px-4 py-2 text-xs text-ink-muted">
                        {formatUnixTimestamp(c.modified_at_unix)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
