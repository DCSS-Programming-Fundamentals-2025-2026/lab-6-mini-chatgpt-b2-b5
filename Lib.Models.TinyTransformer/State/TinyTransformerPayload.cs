using Lib.Models.TinyTransformer.Configuration;
namespace Lib.Models.TinyTransformer.State;

public class TinyTransformerPayload
{
    public TinyTransformerConfig Config { get; set; }
    
    public float[][] TokenEmbeddings { get; set; }
    public float[][] Wq { get; set; }
    public float[][] Wk { get; set; }
    public float[][] Wv { get; set; }
    public float[][] Wo { get; set; }
    public float[][] Ffn1 { get; set; }
    public float[] Ffn1Bias { get; set; }
    public float[][] Ffn2 { get; set; }
    public float[] Ffn2Bias { get; set; }
    public float[][] OutputW { get; set; }
    public float[] OutputBias { get; set; }
}