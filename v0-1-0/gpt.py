
from znet.autograd import Tensor
import znet.nn as nn
import znet
import numpy as np
with open ('input.txt' , 'r' , encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("".join(chars))
print(vocab_size)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# print(encode("hii there"))

# print(decode(encode("hii there")))

data = Tensor(encode(text), requires_grad=False)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 8



print(train_data[:block_size+1])

x = train_data[:block_size]
y =  train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is: {target}")

np.random.seed(1337)
batch_size = 4 # how many independent sequence will we process in parallel
block_size = 8 # What is the maximum context length for predictions?

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = np.stack([data.data[i:i+block_size] for i in ix])
    y = np.stack([data.data[i+1:i+block_size+1] for i in ix])
    return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)
xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("targets: ")
print(yb.shape) 
print(yb)

print('--------------------')
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b,:t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target is: {target}")

        
 
criterion = nn.CrossEntropyLoss(reduction="mean")
from znet.autograd.ops_softmax import softmax
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets = None):
        
        # idx and targets are both  (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = criterion(logits, targets)
        return logits , loss
    def generate(self,idx,max_new_token):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_token):
            # get the prediction
            logits , loss = self(idx)
            # focus only on the last time step
            logits = logits[:,-1,:] # become (B,C)
            # apply softmax to get probabilities
            probs = softmax(logits,axis = -1) # (B ,1)
            # sample from the dictribution
            u   = np.random.rand(probs.shape[0], 1)          # (B, 1)
            cdf = np.cumsum(probs.data, axis=1)                   # (B, C)
            idx_next = (cdf > u).argmax(axis=1).reshape(-1, 1).astype(np.int64)  # (B, 1)
            # append sample index to the running sequence
            idx = np.concatenate((idx,idx_next) , axis = 1) # (B , T+1)
            # print(idx.shape)
        return idx

model = BigramLanguageModel(vocab_size=vocab_size)

logits, loss = model(xb,yb)
print(logits.shape)
print(loss)






# gnerate the new text, it is going to be similar to previous examples:
idx = np.zeros((1,1))
print(decode(model.generate(idx, max_new_token=100)[0].tolist()))



# Training the embedding 
# batch_size = 32
# learning_rate = 1e-2
# eval_iter = 200
# max_iter = 30000
# eval_interval = 300
# import znet.optim as optim
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
# def estimate_loss():
#     out = {}
    
#     for split in ["train" , "val"]:
#         losses = np.zeros(eval_iter)
#         for k in range(eval_iter):
#             X, Y = get_batch(split)
#             _, loss = model(X,Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
    
#     return out

# for iter in range(max_iter):
    
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f} , val loss {losses['val']:.4f}")

#     xb, yb = get_batch('train')
#     logits , loss = model (xb, yb)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# # print(loss.item())

# idx = np.zeros((1,1))
# print(decode(model.generate(idx, max_new_token=100)[0].tolist()))



# Let us implement the single head self attention layer
# using this idea and the idea of key, query, and value

B, T, C = 4, 8, 2
x = np.random.randn(B, T, C)
x = Tensor(x)
head_size = 16
# head and query
key = nn.Linear(C,head_size, bias=False)
query = nn.Linear(C,head_size, bias=False)

# we need to have another layer
value = nn.Linear(C,head_size, bias=False)

k = key (x) # (B,T, 16)
q = query(x) # (B,T, 16)

wei = q @ k.transpose(-2,-1) # (B, T, 16) @ (B, 16, T) -> (B, T, T)

# if our wei would be a realy high numbers, softmax output moves 
# toward a one_hot encoding i.e. only zero and one. So we need to normalize the wei
#you need to normalize by the head_size
wei = wei * (head_size**(-0.5))



tril = np.tri(T, T, dtype=bool)            # (T, T)
wei = wei.masked_fill(~tril, -np.inf)      # or (tril == 0)

wei = softmax(wei,axis = -1)
v = value(x)
out = wei @ v
# now if you look at wei it is not uniformly distributed
# out.backward()
# print(x.grad)
# print(wei)



class Head(nn.Module):
    # one head self attention
    def __init__(self, head_size ):
        super().__init__()
        self.key = nn.Linear(n_embed , head_size,bias=False)
        self.query = nn.Linear(n_embed , head_size,bias=False)
        self.value = nn.Linear(n_embed , head_size,bias=False)
        self.tril = Tensor(np.tri(block_size, block_size, dtype=bool), requires_grad=False)

        self.drouput = nn.Dropout(dropout)
    def forward (self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute the attention score ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5  # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        
        T = x.shape[1]
        # wei = wei.masked_fill(~self.tril[:T, :T], -np.inf)
        
        
        tril = np.tri(T, T, dtype=bool)  
        wei = wei.masked_fill(~tril, -np.inf)      # or (tril == 0)

        wei = softmax(wei,axis = -1)
        wei = self.drouput(wei)
        v = self.value(x)
        out = wei @ v
        return out
import znet.functional as F   
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.drouput = nn.Dropout(dropout)
    def forward(self,x):
        
        out = F.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.drouput(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            # a projection layer similar to Multihead
            nn.Linear(4*n_embed,n_embed),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    # Transformer block: communicationa followed by computation
    def __init__(self, n_embed , n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self,x):
        # add residual here

        # Layer norm is added before the layer unlike the original paper
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.sa_head = Head(n_embed) # one head with n_embed (32-dimensional) size
        # self.sa_heads = MultiHeadAttention(4, n_embed // 4) # i.e. 4 heads of 8-dimensional self attention
        
        # self.ffwd = FeedForward(n_embed)
        # self.blocks = nn.Sequential(
        #     Block(n_embed , 4),
        #     Block(n_embed , 4),
        #     Block(n_embed , 4),
        #     # we need a layer norm here
        #     nn.LayerNorm(n_embed),
        # )
        self.blocks = nn.Sequential(*[Block(n_embed,n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed ,vocab_size)

    def forward(self,idx,targets = None):
        B, T = idx.shape
        # idx and targets are both  (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B,T,n_embed)
        pos_ids = np.arange(T, dtype=np.int64)           # (T,)
        pos_embed = self.position_embedding_table(pos_ids)  # (T, C)
        
        x = token_embed + pos_embed
        #x = self.sa_head(x)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = criterion(logits, targets)
        return logits , loss
    def generate(self,idx,max_new_token):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_token):
            idx_cond = idx[:,-block_size:]
            logits , loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # become (B,C)
            # apply softmax to get probabilities
            probs = softmax(logits,axis = -1) # (B ,1)
            # sample from the dictribution
            u   = np.random.rand(probs.shape[0], 1)          # (B, 1)
            cdf = np.cumsum(probs.data, axis=1)                   # (B, C)
            idx_next = (cdf > u).argmax(axis=1).reshape(-1, 1).astype(np.int64)  # (B, 1)
            # append sample index to the running sequence
            idx = np.concatenate((idx,idx_next) , axis = 1) # (B , T+1)
            # print(idx.shape)
        return idx
    
    
n_embed = 32
n_head = 2
n_layer = 2
dropout = 0.2
block_size = 16
model = BigramLanguageModel(vocab_size=vocab_size)

logits, loss = model(xb,yb)
print(logits.shape)
print(loss)


batch_size = 32

lr = 1e-3 # we have to reduce the learning rate, attention head can not tolerate a high lr
eval_iter = 200
max_iter = 10000
eval_interval = 100
import znet.optim as optim
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)






def estimate_loss():
    out = {}
    model.eval()
    for split in ["train" , "val"]:
        losses = np.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            _, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iter):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f} , val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits , loss = model (xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(loss.item())

# torch.save(model.state_dict(), 'model_weights.pth')

idx = np.zeros((1,1))
print(decode(model.generate(idx, max_new_token=1000)[0].tolist()))


# print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")
# for p in model.parameters():
#     print(p.numel())