--- Number figure captions in the formats that do not do it themselves.
--
-- LaTeX numbers floats, so its captions already read "Figure 3: …". HTML,
-- EPUB and DOCX do not, which left the same caption unnumbered in half the
-- outputs and made a cross-reference in the prose unresolvable. This prefixes
-- the caption in exactly those formats, in document order.
--
-- Hardcoding the number into the Markdown was the other option and is worse:
-- LaTeX would then render "Figure 3: Figure 3. …", and every inserted figure
-- would silently renumber the ones after it.

local n = 0

if FORMAT:match("latex") or FORMAT:match("beamer") then
  return {}
end

return {
  {
    Figure = function(fig)
      n = n + 1
      local caption = fig.caption.long
      if #caption == 0 then return nil end
      local first = caption[1]
      if first.t == "Plain" or first.t == "Para" then
        table.insert(first.content, 1, pandoc.Str("Figure " .. n .. "."))
        table.insert(first.content, 2, pandoc.Space())
      end
      return fig
    end,
  },
}
