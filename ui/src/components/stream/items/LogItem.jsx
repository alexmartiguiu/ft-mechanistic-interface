// Raw streamed log lines (when replaying a run's training log).
export default function LogItem({ item }) {
  return (
    <div className="si-log">
      {item.lines.map((l, i) => (
        <div key={i}>{l}</div>
      ))}
    </div>
  );
}
