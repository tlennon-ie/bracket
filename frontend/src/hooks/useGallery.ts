import { api } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";

export function useGallery() {
	return useQuery({
		queryKey: ["gallery"],
		queryFn: api.getGallery,
		staleTime: 5_000,
		refetchInterval: (q) =>
			q.state.data && q.state.data.length > 0 ? 10_000 : 5_000,
	});
}
