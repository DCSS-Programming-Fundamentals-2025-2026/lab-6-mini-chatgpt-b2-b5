namespace Lib.Batching.Tests;

public class ArrayTokenStream : ITokenStream
{
    private readonly int[] _tokens;

    public ArrayTokenStream(int[] tokens)
    {
        _tokens = tokens;
    }

    public int[] GetTokens()
    {
        return _tokens;
    }
}