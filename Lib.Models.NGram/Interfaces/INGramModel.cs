public interface INGramModel
{
    public void Train(ReadOnlySpan<int> tokens);
}