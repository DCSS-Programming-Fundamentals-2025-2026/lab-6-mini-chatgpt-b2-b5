namespace Lib.Batching;

public class Batch
{
    public int[][] Inputs { get; }
    public int[] Targets { get; }

    public Batch(int[][] inputs, int[] targets)
    {
        Inputs = inputs;
        Targets = targets;
    }
}