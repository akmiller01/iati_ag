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
        idfutils.idf[values[1]] = {index = wordCount, value = tonumber(values[2])}
    end
end

idfutils.wordCount = wordCount


idfutils.tfidf = function (self,sentence)
    local tfidfVec = torch.Tensor(self.wordCount):zero()
    local sentenceWords = sentence:splitAtSpaces()
	local sentenceWordsLen = #sentenceWords
    for j=1,sentenceWordsLen,1 do
		local lowerword = string.lower(sentenceWords[j])
        --Find whether word exists in idf table
        if self.idf[lowerword] ~= nil then
            local index = self.idf[lowerword].index
            local idf = self.idf[lowerword].value
            --Calc tf
            local tf = 0
            for word in string.gmatch(string.lower(sentence),lowerword) do
                    tf = tf + 1
            end
			tf = tf / sentenceWordsLen
            --place tf/idf at correct index in tfidfVec
            tfidfVec[index] = tf/idf
        end
    end
    return tfidfVec
end

return idfutils
