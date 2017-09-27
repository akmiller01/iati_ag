require 'torch'
require 'nn'
idfutils = require 'idfutils'

-- Command line parameters
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options for my NN')
cmd:option('-csv',"/media/alex/Windows/git/iati_ag/to_classify.csv",'csv file')
cmd:option('-model',"tfidf_model.th",'prebuilt model')
cmd:option('-header',true,'csv has header')
-- etc...
cmd:text()
opt = cmd:parse(arg)

-- Data requirement: lua table with method size()
-- Method to read CSV
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

function loadData(dataFile,header)
    local dataset = {}
    local length = 0
    local i = 1
    for line in io.lines(dataFile) do
        if header == true then
            header = false
        else
            local values = line:splitAtCommas()
            local y = torch.Tensor(1)
            y[1] = values[1] -- the target class is the fist number in the line
            local x = torch.Tensor(idfutils.wordCount):zero() -- Initialize a tensor of zeroes
            x = idfutils:tfidf(values[2]) --Pass whole sentence to tfidf vectorizer
            dataset[i] = {x, y}
            i = i + 1
        end
    end
    function dataset:size() return (i - 1) end -- the requirement mentioned
    function dataset:length() return idfutils.wordCount end
    return dataset
end

classifySet = loadData(opt.csv,opt.header)

-- Classify
mlp = torch.load(opt.model)

out = assert(io.open("./prediction_vectors_tfidf.csv", "w"))

for i = 1, classifySet:size(), 1 do
    local x = classifySet[i][1]
    local predictions = mlp:forward(x)
    for j=1,predictions:size(1) do
        out:write(predictions[j])
        if j~=predictions:size(1) then
            out:write(",")
        end
    end
    out:write("\n")
end

out:close()
