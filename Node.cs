using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class NodeExt{
            public string Id{get;set;}
            public int Row;
            public int Col;
            public int Value;
            public int distanceFromStart; //G
            public int estimatedDistanceToEnd; //F

            public NodeExt? CameFrom;

            public NodeExt(int row, int col, int value){

                this.Id = row.ToString() +"-"+col.ToString();
                this.Row= row;
                this.Col = col;
                this.Value= value;
                this.distanceFromStart = Int32.MaxValue;
                this.estimatedDistanceToEnd = Int32.MaxValue;
                this.CameFrom= null;
            }
        }


}