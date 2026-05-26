export default function MatchPage({ params }: { params: { id: string } }) {
  return (
    <>
      <h1 className="text-xl font-bold mb-2">Match {params.id}</h1>
      <p className="text-neutral-400 text-sm">
        TODO: probability split, expected goals, score-distribution heatmap, related Polymarket markets.
      </p>
    </>
  )
}
