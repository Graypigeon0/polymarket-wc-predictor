export default async function TeamPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  return (
    <>
      <h1 className="text-xl font-bold mb-2">Team {id}</h1>
      <p className="text-neutral-400 text-sm">
        TODO: current ratings, 26-man squad with starter probs, active rating deltas
        with news audit trail, tournament-outright probability over time.
      </p>
    </>
  )
}
