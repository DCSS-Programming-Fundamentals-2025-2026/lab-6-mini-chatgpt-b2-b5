namespace Contracts;

public interface ITextGenerator
{
    string Generate(string prompt, int maxTokens,
    float temperature, int topK, int? seed = null);
}