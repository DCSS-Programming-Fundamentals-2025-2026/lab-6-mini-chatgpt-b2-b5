using Contracts;

public interface INGramModelFactory
{
    ILanguageModel Create(string modelType, int vocabSize);
}