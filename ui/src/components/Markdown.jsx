import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/* Inline markdown for the agent stream. The producers emit prose with markdown
   emphasis (**bold**, *italic*, `code`, [links](…), ~~strike~~); this renders it
   while STRIPPING block wrappers (<p>/<ul>/<li>) so it drops straight into an
   existing <p>/<li>/<span> and inherits that element's styling untouched.
   react-markdown escapes raw HTML, so this is XSS-safe. */
const COMPONENTS = {
  p: ({ children }) => <>{children}</>,
  ul: ({ children }) => <>{children}</>,
  ol: ({ children }) => <>{children}</>,
  li: ({ children }) => <>{children}</>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
  ),
};

export default function Markdown({ children }) {
  if (children == null || children === "") return null;
  const text = typeof children === "string" ? children : String(children);
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
      {text}
    </ReactMarkdown>
  );
}
