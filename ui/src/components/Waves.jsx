/* A thin "waterline" of drifting wavelets for the landing header. The path tiles
   at a 300-unit period and is drawn one period wider than the viewBox on each
   side, so the CSS translateX(-300) loop reads as continuous water. Two layers at
   different speeds/opacities give a touch of parallax. Purely decorative. */
const waveD = (y) => {
  let d = `M -300 ${y}`;
  for (let i = 0; i < 7; i++) d += " q 75 -5 150 0 q 75 5 150 0";   // crest then trough = one 300px period
  return d;
};

export default function Waves() {
  return (
    <div className="waves" aria-hidden="true">
      <svg viewBox="0 0 1500 24" preserveAspectRatio="none">
        <path className="wave wave-back" d={waveD(16)} />
        <path className="wave wave-front" d={waveD(12)} />
      </svg>
    </div>
  );
}
