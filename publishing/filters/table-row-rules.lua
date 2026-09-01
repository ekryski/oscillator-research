--- Draw a rule between table body rows, in the LaTeX outputs only.
--
-- These tables carry cells that wrap to five or six lines, and booktabs rules
-- only the head and the foot, so where one row ends and the next begins is
-- left to the reader to infer from column alignment. The HTML outputs already
-- separate rows, through `tbody tr + tr td` in publishing/css/tables.css; this
-- is the same decision for the PDF.
--
-- `\cmidrule` is legal at the start of a row and is lighter than `\midrule`,
-- which reads as a section break rather than a row boundary. Prepending it to
-- the first cell of every row after the first puts it exactly where LaTeX
-- wants it, immediately after the preceding row's `\\`.

if not FORMAT:match("latex") then
  return {}
end

--- Prepend a raw-LaTeX rule to every row of `rows` except the first.
local function rule_between(rows, ncols)
  local rule = pandoc.RawInline("tex", "\\cmidrule{1-" .. ncols .. "}")
  for i = 2, #rows do
    local cell = rows[i].cells[1]
    local block = cell and cell.contents[1]
    -- Plain and Para hold inlines; a cell holding a list or a nested table has
    -- nowhere to put an inline, and is left alone rather than mangled
    if block and block.content then
      table.insert(block.content, 1, rule)
    end
  end
end

return {
  {
    Table = function(tbl)
      local ncols = #tbl.colspecs
      for _, body in ipairs(tbl.bodies) do
        rule_between(body.body, ncols)
      end
      return tbl
    end,
  },
}
