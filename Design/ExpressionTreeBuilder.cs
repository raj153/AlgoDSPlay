using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1628. Design an Expression Tree With Evaluate Function	
    https://leetcode.com/problems/design-an-expression-tree-with-evaluate-function/description/
    
    Complexity:
    Time complexity:O(n)
    Space complexity:O(n)

    */
    abstract class ExpressionNode
    {
        public abstract int Evaluate();

        //Factory method pattern
        public static ExpressionNode From(string value)
        {
            switch (value)
            {
                case "+":
                    return new AdditionNode();
                case "-":
                    return new SubtractionNode();
                case "*":
                    return new MultiplicationNode();
                case "/":
                    return new DivisionNode();
                default:
                    return new NumericalNode(value);
            }
        }
    }

    abstract class OperatorNode : ExpressionNode
    {
        protected ExpressionNode leftNode;
        protected ExpressionNode rightNode;

        public void SetLeft(ExpressionNode left)
        {
            this.leftNode = left;
        }

        public void SetRight(ExpressionNode right)
        {
            this.rightNode = right;
        }
    }

    class AdditionNode : OperatorNode
    {
        public override int Evaluate()
        {
            return leftNode.Evaluate() + rightNode.Evaluate();
        }
    }

    class SubtractionNode : OperatorNode
    {
        public override int Evaluate()
        {
            return leftNode.Evaluate() - rightNode.Evaluate();
        }
    }

    class MultiplicationNode : OperatorNode
    {
        public override int Evaluate()
        {
            return leftNode.Evaluate() * rightNode.Evaluate();
        }
    }

    class DivisionNode : OperatorNode
    {
        public override int Evaluate()
        {
            return leftNode.Evaluate() / rightNode.Evaluate();
        }
    }

    class NumericalNode : ExpressionNode
    {
        private string numericValue;

        public NumericalNode(string value)
        {
            numericValue = value;
        }

        public override int Evaluate()
        {
            return int.Parse(numericValue);
        }
    }

    class ExpressionTreeBuilder
    {
        public ExpressionNode BuildTree(string[] postfix)
        {
            Stack<ExpressionNode> stack = new Stack<ExpressionNode>();

            foreach (string s in postfix)
            {
                ExpressionNode node = ExpressionNode.From(s);
                if (node is NumericalNode)
                {
                    stack.Push(node);
                }
                else if (node is OperatorNode)
                {
                    OperatorNode operatorNode = (OperatorNode)node;
                    operatorNode.SetRight(stack.Pop());
                    operatorNode.SetLeft(stack.Pop());
                    stack.Push(operatorNode);
                }
                else
                {
                    throw new InvalidOperationException("Node should be instance of NumericalNode or OperatorNode");
                }
            }

            return stack.Pop();
        }
    }
    /**
 * Your TreeBuilder object will be instantiated and called as such:
 * ExpressionTreeBuilder obj = new ExpressionTreeBuilder();
 * Node expTree = obj.buildTree(postfix);
 * int ans = expTree.evaluate();
 */

}
