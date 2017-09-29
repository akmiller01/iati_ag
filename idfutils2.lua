torch.setdefaulttensortype('torch.FloatTensor')

opt = {
	csvfilename = 'train.csv',
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
local docCount = 0
local wordCount = 0
for line in io.lines(opt.csvfilename) do
    if header == true then
        header = false
    else
		local values = line:splitAtCommas()
        docCount = docCount + 1
        words = values[2]:splitAtSpaces()
		sentenceWords = {}
		for j=1,#words,1 do
			local lowerWord = string.lower(words[j])
			if idfutils.idf[lowerWord] == nil then-- new word
				wordCount = wordCount + 1
				idfutils.idf[lowerWord] = {index = wordCount, value = 1}
				sentenceWords[lowerWord] = 1
			else -- Already appeared in another document
				if sentenceWords[lowerWord] == nil then-- But not in this document
					idfutils.idf[lowerWord].value = idfutils.idf[lowerWord].value + 1
					sentenceWords[lowerWord] = 1
				end
			end
		end
    end
end

idfutils.wordCount = wordCount

for i=1,#idfutils.idf,1 do
		idfutils.idf[i].value = math.log(docCount/idfutils.idf[i].value)
end

idfutils.wordidf = function(self,word)
	local lowerWord = string.lower(word)
	if idfutils.idf[lowerWord] == nil then
		return 1
	else
		return idfutils.idf[lowerWord].value
	end
end


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
            tfidfVec[index] = tf*idf
        end
    end
    return tfidfVec
end

idfutils.save = function(self,destination)
	-- Save for later
	torch.save(destination, self)
end

return idfutils
