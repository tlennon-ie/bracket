import { api } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";

export function useFamilies() {
	return useQuery({
		queryKey: ["presets", "families"],
		queryFn: api.listFamilies,
		staleTime: 5 * 60_000,
	});
}

export function useTrainingTypes(family: string | null) {
	return useQuery({
		queryKey: ["presets", "types", family],
		queryFn: () => api.listTrainingTypes(family as string),
		enabled: !!family,
		staleTime: 5 * 60_000,
	});
}

export function usePreset(family: string | null, type: string | null) {
	return useQuery({
		queryKey: ["presets", "spec", family, type],
		queryFn: () => api.getPreset(family as string, type as string),
		enabled: !!family && !!type,
		staleTime: 5 * 60_000,
	});
}
