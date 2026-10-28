"""Step 1 (3.3): attention with no trainable weights. Scores are just dot products of the inputs with each other."""
import torch

WORDS = ["Your", "journey", "starts", "with", "one", "step"]
INPUTS = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)


def simplified_attention(inputs):
    scores = inputs @ inputs.T               # (T, T): how similar every token is to every other token
    weights = torch.softmax(scores, dim=-1)  # each row sums to 1
    return weights, weights @ inputs         # context vector = weighted sum of all inputs


if __name__ == "__main__":
    # one query by hand, the way the book builds it up
    query = INPUTS[1]
    scores = torch.tensor([torch.dot(x, query) for x in INPUTS])
    print(f"scores for 'journey'  : {scores.numpy().round(4)}")
    print(f"softmax               : {torch.softmax(scores, 0).numpy().round(4)}")

    # then all queries at once with one matmul
    weights, context = simplified_attention(INPUTS)
    print(f"\nweights = softmax(inputs @ inputs.T)   shape={tuple(weights.shape)}")
    print(" " * 9 + "".join(f"{w:>9}" for w in WORDS))
    for w, row in zip(WORDS, weights):
        print(f"{w:>9}" + "".join(f"{v:9.3f}" for v in row))
    print(f"\ncontext = weights @ inputs   shape={tuple(context.shape)}")
    print(context.numpy().round(4))
    print("\nNo trainable weights, so 'journey' can only attend by raw similarity. That's what 3.4 fixes.")
