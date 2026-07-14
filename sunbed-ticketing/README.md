# Sunbed Ticketing

A single-page, offline-friendly grid tool for tracking a 6×10 layout of beach umbrellas — no login, no server, no build step.

## Use it

Open `index.html` directly in any phone/tablet/desktop browser (double-click it, or host the folder as static files). To keep it handy on a phone, open it once in the browser then "Add to Home Screen."

## How it works

- Umbrellas are labeled by row/column, e.g. `A1` (row A, seat 1) through `F10`.
- **Tap a free (green) umbrella** to check it in — set guest name, party size, price, and notes, and mark it Occupied or Reserved.
- **Tap an occupied/reserved umbrella** to view its details, edit them, check it out (frees it up), or move it.
- **Move**: tap Move, then tap any free umbrella to relocate the booking there (all its details come with it). Tap the source umbrella again, or "Cancel," to back out.
- **Check out** clears an umbrella back to free — used instead of a hard delete so nothing is lost by accident (confirmation prompt included).
- **Export / Import** at the bottom saves/restores the whole grid as a JSON file — useful as a backup or to hand off a shift.
- **Clear all** resets every umbrella to free (confirmation prompt included).

## Data storage

Everything is saved in the browser's `localStorage` on the device you're using — it persists across refreshes and closing the tab, but it is **local to that one browser/device**. It does not sync between multiple phones/tablets. If you need several staff devices to see the same live grid, that would need a small shared backend — let me know and I can add that separately.
