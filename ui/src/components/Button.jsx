export default function Button({ variant = "default", size, children, ...rest }) {
  const cls = ["btn", variant !== "default" ? variant : "", size === "sm" ? "sm" : ""]
    .filter(Boolean).join(" ");
  return (
    <button className={cls} {...rest}>
      {children}
    </button>
  );
}
