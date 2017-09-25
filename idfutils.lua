torch.setdefaulttensortype('torch.FloatTensor')

opt = {
	csvfilename = 'idf.csv',
}
local idfutils = {}
idfutils.idf = {}

function string:splitAtCommas()
    local sep, values = ",", {}
    local pattern = string.format("([^%s]+)", sep)
    self:gsub(pattern, function(c) values[#values+1] = c end)
    return values
end

function string:splitAtSpaces()
    local sep, values = " ", {}
    local pattern = string.format("([^%s]+)", sep)
    self:gsub(pattern, function(c) values[#values+1] = c end)
    return values
end

local header = true
local wordCount = 0
for line in io.lines(opt.csvfilename) do
    if header == true then
        header = false
    else
        wordCount = wordCount + 1
        local values = line:splitAtCommas()
        idfutils.idf[values[1]] = {index = wordCount, value = values[2]}
    end
end


idfutils.tfidf = function (self,sentence)
    local tfidfVec = torch.Tensor(wordCount)
    local sentenceWords = sentence:splitAtSpaces()
    for j=1,#sentenceWords,1 do
        --Find whether word exists in idf table
        if self.idf[sentenceWords[j]] ~= nil then
            local index = self.idf[sentenceWords[j]].index
            local idf = self.idf[sentenceWords[j]].idf
            --Calc tf
            local tf = 0
            for word in string.gmatch(sentence,sentenceWords[j]) do
                    tf = tf + 1
            end
            --place tf/idf at correct index in tfidfVec
            tfidfVec[index] = tf/idf
        end
    end
    return tfidfVec
end

return idfutils
