export default function TeamPage({ params }: { params: { id: string } }) {
  return (
    <>
      <h1 className="text-xl font-bold mb-2">Team {params.id}</h1>
      <p className="text-neutral-400 text-sm">
        TODO: current ratings, 26-man squad with starter probs, active rating deltas
        with news audit trail, tournament-outright probability over time.
      </p>
    </>
  )
}
