export default async function MatchPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  return (
    <>
      <h1 className="text-xl font-bold mb-2">Match {id}</h1>
      <p className="text-neutral-400 text-sm">
        TODO: probability split, expected goals, score-distribution heatmap, related Polymarket markets.
      </p>
    </>
  )
}
