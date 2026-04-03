namespace Contracts;

public interface ITokenizer
{
    int VocabSize { get; }
    int[] Encode(string text); 
    string Decode(ReadOnlySpan<int> tokens);
    object GetPayloadForCheckpoint(); 
}
