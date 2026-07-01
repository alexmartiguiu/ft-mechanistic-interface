import { motion } from "motion/react";

/* An iOS-style on/off switch. The knob slides with a spring (motion `layout`); the
   track colour crossfades via CSS. Used to flip the left panel into developer mode. */
export default function Switch({ checked, onChange, label, disabled = false }) {
  return (
    <button
      type="button"
      className={`ios-switch ${checked ? "on" : ""}`}
      role="switch"
      aria-checked={checked}
      aria-label={label || "toggle"}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
    >
      {label && <span className="ios-switch-label">{label}</span>}
      <span className="ios-track" aria-hidden="true">
        <motion.span
          className="ios-knob"
          layout
          transition={{ type: "spring", stiffness: 550, damping: 32, mass: 0.7 }}
        />
      </span>
    </button>
  );
}
