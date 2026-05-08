import { ApiError, api } from "@/lib/api";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

export function useReport() {
	return useQuery({
		queryKey: ["report"],
		queryFn: async () => {
			try {
				return await api.getReport();
			} catch (e) {
				if (e instanceof ApiError && e.status === 404) {
					return null;
				}
				throw e;
			}
		},
		staleTime: 10_000,
	});
}

export function useRegenerateReport() {
	const qc = useQueryClient();
	return useMutation({
		mutationFn: api.regenerateReport,
		onSuccess: (data) => {
			qc.setQueryData(["report"], data.content);
			toast.success("Report regenerated", {
				description: data.path,
			});
		},
	});
}
