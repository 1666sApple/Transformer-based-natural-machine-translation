class BahdanauEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim, dropout_p):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, encoder_hidden_dim,
        bidirectional=True)
        self.linear = nn.Linear(encoder_hidden_dim * 2,
        decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)