--- Number figure and table captions in the formats that do not do it themselves.
--
-- LaTeX numbers floats, so its captions already read "Figure 3: …" and
-- "Table 1: …". HTML, EPUB and DOCX do not, which left the same caption
-- unnumbered in half the outputs and made a cross-reference in the prose
-- unresolvable: the manuscript says "every paper in Table 1" four times, and
-- an HTML reader had no table labelled 1 to look for.
--
-- Figures and tables carry independent counters, matching LaTeX, and both
-- count in document order.
--
-- Hardcoding the number into the Markdown was the other option and is worse:
-- LaTeX would then render "Figure 3: Figure 3. …", and every inserted float
-- would silently renumber the ones after it.

local counts = { Figure = 0, Table = 0 }

if FORMAT:match("latex") or FORMAT:match("beamer") then
  return {}
end

--- Prefix a float's caption with "<label> <n>." in place.
--
-- Returns nil for a float with no caption, which leaves it untouched: an
-- uncaptioned float has nowhere to put the number, and LaTeX does not number
-- one either, so the two stay in step.
local function prefix(float, label)
  local caption = float.caption.long
  if #caption == 0 then return nil end
  local first = caption[1]
  if first.t ~= "Plain" and first.t ~= "Para" then return nil end
  counts[label] = counts[label] + 1
  table.insert(first.content, 1, pandoc.Str(label .. " " .. counts[label] .. "."))
  table.insert(first.content, 2, pandoc.Space())
  return float
end

return {
  {
    Figure = function(fig) return prefix(fig, "Figure") end,
    Table = function(tbl) return prefix(tbl, "Table") end,
  },
}
