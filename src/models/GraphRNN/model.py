import torch
import torch.nn as nn

class GraphLevelRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=4):
        super().__init__()
        self.input_embed=nn.Sequential(nn.Linear(input_size, embedding_size), 
        nn.ReLU())
        self.rnn=nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size=hidden_size
        self.num_layers=num_layers

    def forward(self, x, lengths, hidden=None):
        x=self.input_embed(x)
        packed=nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        ) 
        out, hidden=self.rnn(packed, hidden)
        out, _=nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out, hidden

    def step(self, x, hidden):
        x=self.input_embed(x)
        out, hidden=self.rnn(x, hidden)
        return out, hidden


class EdgeLevelRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers=4):
        super().__init__()
        self.input_embed=nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.ReLU()
        )
        self.rnn=nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.output=nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )
        self.hidden_size=hidden_size
        self.num_layers=num_layers

    def init_hidden(self, graph_hidden, proj):
        h0=proj(graph_hidden)
        return h0.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()

    def forward(self, x, hidden):
        x=self.input_emnbed(x)
        out, hidden=self.rnn(x, hidden)
        logits=self.output(out).squeeze(-1)
        return torch.sigmoid(logits), hidden


class EdgeLevelMLP(nn.Module):
    def __init__(self, hidden_size, embedding_size, max_prev_node):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, max_prev_node)
        )

    def forward(self, graph_hidden):
        return torch.sigmoid(self.mlp(graph_hidden))


#PUT ALL TOGETHER
class GraphRNN(nn.Module):
    def __init__(self, max_prev_node, graph_hidden=128, graph_embed=64, edge_hidden=16,
    edge_embed=8, graph_layers=4, edge_layers=4, variant="rnn"):
        super().__init__()
        assert variant in ("rnn", "s")
        self.M=max_prev_node
        self.variant=variant
        self.graph_rnn=GraphLevelRNN(max_prev_node, graph_embed, graph_hidden, graph_layers)

        if variant=="rnn":
            self.edge_rnn=EdgeLevelRNN(edge_embed, edge_hidden, edge_layers)
            self.proj=nn.Linear(graph_hidden, edge_hidden)
        else:
            self.edge_mlp=EdgeLevelMLP(graph_hidden, graph_embed, max_prev_node)

    #Training
    def forward(self, x, y, lengths):
        graph_out, _=self.graph_rnn(x, lengths)
        B, T, H=graph_out.shape
        if self.variant=="s":
            return self.edge_mlp(graph_out)

        flat_hidden=graph_out.reshape(B*T, H)
        flat_target=y.reshape(B*T, self.M)

        edge_in=torch.zeros_like(flat_target) 
        edge_in[:, 0]=1.0
        edge_in[:, 1:]=flat_target[:, :-1]
        edge_in=edge_in.unsqueeze(-1)

        h0=self.edge_rnn.init_hidden(flat_hidden, self.proj)
        pred, _=self.edge_rnn(edge_in, h0)
        return pred.reshape(B, T, self.M)

    #Generation
    @torch.no_grad()
    def generate(self, num_graphs, max_num_node, device="cpu"):
        self.eval()
        results=[]
        for _ in range(num_graphs):
            S=torch.zeros(max_num_node -1, self.M)
            x_t=torch.ones(1,1, self.M, device=device)
            h=None
            steps_used=0
            for t in range(max_num_node-1):
                graph_out, h=self.graph_rnn.step(x_t, h)
                graph_out=graph_out.squeeze(1)
                if self.variant=="s":
                    probs=self.edge_mlp(graph_out)
                    sample=torch.bernoulli(probs)
                else:
                    sample=torch.zeros(1, self.M, device=device)
                    he=self.edge_rnn.init_hidden(graph_out, self.proj)
                    edge_in=torch.ones(1, 1, 1, device=device)
                    valid=min(t+1, self.M)
                    for k in range(valid):
                        prob, he=self.edge_rnn(edge_in, he)
                        bit=torch.bernoulli(prob)
                        sample[0,k]=bit[0, 0]
                        edge_in=bit.view(1, 1, 1)
                S[t]=sample[0].cpu()
                steps_used=t+1
                if sample.sum().item()==0 and t>0:
                    break
                x_t=sample.view(1, 1, self.M)
            results.append(S[:steps_used].numpy())
        return results
        


        