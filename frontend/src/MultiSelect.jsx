import React, { useState, useRef, useEffect } from "react";

/**
 * A dropdown that shows checkboxes for each option.
 *
 * Props:
 *  - label       (string)   – field label
 *  - options     (string[]) – all possible values
 *  - selected    (string[]) – currently selected values
 *  - onChange    (string[] => void) – called with new selection
 */
export default function MultiSelect({ label, options, selected, onChange }) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef(null);

  // close on outside click
  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const toggle = (val) => {
    const next = selected.includes(val)
      ? selected.filter((v) => v !== val)
      : [...selected, val];
    if (next.length > 0) onChange(next);
  };

  const selectAll = () => onChange([...options]);
  const clearAll = () => onChange([options[0]]); // keep at least one

  const filtered = search
    ? options.filter((o) => o.toLowerCase().includes(search.toLowerCase()))
    : options;

  const summary =
    selected.length === 0
      ? "None"
      : selected.length === 1
      ? selected[0]
      : selected.length === options.length
      ? "All selected"
      : `${selected.length} selected`;

  return (
    <div className="ms" ref={ref}>
      <span className="ms-label">{label}</span>
      <button
        type="button"
        className={`ms-trigger ${open ? "ms-open" : ""}`}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="ms-summary">{summary}</span>
        <span className="ms-arrow">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="ms-dropdown">
          {options.length > 6 && (
            <input
              className="ms-search"
              type="text"
              placeholder="Search…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              autoFocus
            />
          )}
          <div className="ms-actions">
            <button type="button" onClick={selectAll}>All</button>
            <button type="button" onClick={clearAll}>Clear</button>
          </div>
          <ul className="ms-list">
            {filtered.map((opt) => (
              <li key={opt} className="ms-item" onClick={() => toggle(opt)}>
                <input
                  type="checkbox"
                  checked={selected.includes(opt)}
                  readOnly
                  tabIndex={-1}
                />
                <span>{opt}</span>
              </li>
            ))}
            {filtered.length === 0 && (
              <li className="ms-empty">No matches</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
