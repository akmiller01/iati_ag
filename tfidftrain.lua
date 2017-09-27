require 'torch'
require 'nn'
idfutils = require 'idfutils'

-- Command line parameters
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options for my NN')
cmd:option('-units',1000,'units in the hidden layer')
cmd:option('-learningRate',0.1,'learning rate')
cmd:option('-trainCsv',"/media/alex/Windows/git/iati_ag/train.csv",'training csv file')
cmd:option('-testCsv',"/media/alex/Windows/git/iati_ag/test.csv",'testing csv file')
cmd:option('-header',true,'csv has header')
-- etc...
cmd:text()
opt = cmd:parse(arg)


-- Fully connected feed-forward network container
mlp = nn.Sequential()

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

trainingSet = loadData(opt.trainCsv,opt.header)
testingSet = loadData(opt.testCsv,opt.header)
print("Training obs: ",trainingSet:size())
print("Testing obs: ",testingSet:size())

-- Using tanh as transfer function for non-linearlity
inputSize = trainingSet:length()
hiddenLayer1Size = opt.units
hiddenLayer2Size = opt.units

mlp:add(nn.Linear(inputSize,hiddenLayer1Size))
mlp:add(nn.HardTanh())
--mlp:add(nn.Linear(hiddenLayer1Size,hiddenLayer2Size))
--mlp:add(nn.HardTanh())

-- outputs
nclasses = 8

mlp:add(nn.Linear(hiddenLayer1Size,nclasses))
mlp:add(nn.LogSoftMax())

-- Training with SGD and negative log-likelihood criterion

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(mlp,criterion)
trainer.learningRate = opt.learningRate

-- Training!
trainer:train(trainingSet)

-- Save for later
torch.save('tfidf_model.th', mlp)

-- Accuracy
function argmax(v)
  local maxvalue = torch.max(v)
  for i=1,v:size(1) do
    if v[i] == maxvalue then
      return i
    end
  end
end

tot = 0
pos = 0
for i = 1, testingSet:size(), 1 do
    local x = testingSet[i][1]
    local y = testingSet[i][2]
    local prediction = argmax(mlp:forward(x))
    if math.floor(prediction) == math.floor(y[1]) then
        pos = pos + 1
    end
    tot = tot + 1
end
print("Accuracy(%) is " .. pos/tot*100)
