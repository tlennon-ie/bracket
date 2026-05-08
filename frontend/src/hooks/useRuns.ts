import { api } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";

export function useRuns() {
	return useQuery({
		queryKey: ["runs"],
		queryFn: api.listRuns,
		staleTime: 5_000,
	});
}

export function useRunDetail(runId: string | null) {
	return useQuery({
		queryKey: ["run-detail", runId],
		queryFn: () => api.getRun(runId as string),
		enabled: !!runId,
		staleTime: 30_000,
	});
}
