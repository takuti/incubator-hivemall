/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
package hivemall.tools;

import hivemall.evaluation.BinaryResponsesMeasures;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hashing.MurmurHash3;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.typeinfo.ListTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @link https://github.com/prestodb/presto/blob/master/presto-main/src/main/java/com/facebook/presto/operator/aggregation/ApproximateCountDistinctAggregations.java
 */
@Description(
        name = "approx_distinct",
        value = "_FUNC_(primitive x [, const double maxStandardError = 0.023])"
                + " - Returns HitRate")
public final class ApproxDistinctUDAF extends AbstractGenericUDAFResolver {

    // prevent instantiation
    private ApproxDistinctUDAF() {}

    @Override
    public GenericUDAFEvaluator getEvaluator(@Nonnull TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 1 && typeInfo.length != 2) {
            throw new UDFArgumentTypeException(typeInfo.length - 1,
                "_FUNC_ takes one or two arguments");
        }

        if (!HiveUtils.isPrimitiveTypeInfo(typeInfo[0])) {
            throw new UDFArgumentTypeException(0,
                "The first argument `x` must be primitive type, but: " + typeInfo[0]);
        }

        return new ApproxDistinctUDAF.Evaluator();
    }

    public static class Evaluator extends GenericUDAFEvaluator {

        private PrimitiveObjectInspector xOI;
        private PrimitiveObjectInspector maxStandardErrorOI;

        private StructObjectInspector internalMergeOI;
        private StructField countField;

        public Evaluator() {}

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] parameters) throws HiveException {
            assert (parameters.length == 1 || parameters.length == 2) : parameters.length;
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.xOI = (PrimitiveObjectInspector) parameters[0];
                if (parameters.length == 2) {
                    this.maxStandardErrorOI = HiveUtils.asDoubleOI(parameters[1]);
                }
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.countField = soi.getStructFieldRef("count");
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = internalMergeOI();
            } else {// terminate
                outputOI = PrimitiveObjectInspectorFactory.writableLongObjectInspector;
            }
            return outputOI;
        }

        private static StructObjectInspector internalMergeOI() {
            ArrayList<String> fieldNames = new ArrayList<String>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("count");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);

            return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
        }

        @Override
        public HyperLogLogAggregationBuffer getNewAggregationBuffer() throws HiveException {
            HyperLogLogAggregationBuffer hllAggr = new HyperLogLogAggregationBuffer();
            reset(hllAggr);
            return hllAggr;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            HyperLogLogAggregationBuffer hllAggr = (HyperLogLogAggregationBuffer) agg;
            hllAggr.reset();
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            HyperLogLogAggregationBuffer hllAggr = (HyperLogLogAggregationBuffer) agg;

            Object x = xOI.getPrimitiveJavaObject(parameters[0]);

            // same as Presto `approx_distinct`
            double maxStandardError = 0.023d;
            if (parameters.length == 2) {
                maxStandardError = PrimitiveObjectInspectorUtils.getDouble(parameters[1], maxStandardErrorOI);
                if (maxStandardError < 0.0115d || maxStandardError > 0.26d) {
                    throw new UDFArgumentException(
                        "The second argument `double maxStandardError` must be in [0.0115, 0.26]: "
                                + maxStandardError);
                }
            }

            hllAggr.iterate(x, maxStandardError);
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            HyperLogLogAggregationBuffer hllAggr = (HyperLogLogAggregationBuffer) agg;

            Object[] partialResult = new Object[1];
            partialResult[0] = new LongWritable(hllAggr.count);
            return partialResult;
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            Object countObj = internalMergeOI.getStructFieldData(partial, countField);
            long count = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(countObj);

            HyperLogLogAggregationBuffer hllAggr = (HyperLogLogAggregationBuffer) agg;
            hllAggr.merge(count);
        }

        @Override
        public LongWritable terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            HyperLogLogAggregationBuffer hllAggr = (HyperLogLogAggregationBuffer) agg;
            return new LongWritable(hllAggr.get());
        }

    }

    /**
     * Presto's `approx_distinct` uses airlift's HyperLogLog implementation:
     * @link https://github.com/airlift/airlift/blob/5367941b6997c0d5dac3ab7263c88c4a4d002c90/stats/src/main/java/io/airlift/stats/cardinality/HyperLogLog.java
     *
     * SparseHll represents single HyperLogLog instance, and it has `mergeWith` method:
     * @link https://github.com/airlift/airlift/blob/5367941b6997c0d5dac3ab7263c88c4a4d002c90/stats/src/main/java/io/airlift/stats/cardinality/SparseHll.java
     *
     * Murmurhash3-128bits is implemented in airlift's slice package:
     * @link https://github.com/airlift/slice/blob/master/src/main/java/io/airlift/slice/Murmur3Hash128.java
     * @link https://github.com/aappleby/smhasher/blob/61a0530f28277f2e850bfc39600ce61d02b518de/src/MurmurHash3.cpp
     */
    public static final class HyperLogLogAggregationBuffer extends
            GenericUDAFEvaluator.AbstractAggregationBuffer {

        // 6 bits to encode the number of zeros after the truncated hash
        // and be able to fit the encoded value in an integer
        private static final int VALUE_BITS = 6;
        private static final int VALUE_MASK = (1 << VALUE_BITS) - 1;
        private static final int EXTENDED_PREFIX_BITS = Integer.SIZE - VALUE_BITS;

        private int[] entries;
        private byte indexBitLength;

        public HyperLogLogAggregationBuffer() {
            super();
        }

        void reset() {
            this.entries = null;
        }

        void merge(long o_count) {
            this.count += o_count;
        }

        long get() {
            if (count == 0) {
                return 0L;
            }
            return count;
        }

        void iterate(@Nonnull final Object x, final double maxStandardError) {
            if (entries == null) {
                /**
                 * @link https://github.com/prestodb/presto/blob/master/presto-main/src/main/java/com/facebook/presto/operator/aggregation/ApproximateCountDistinctAggregations.java#L99-L108
                 */
                int numberOfBuckets = log2Ceiling((int) Math.ceil(1.0816 / (maxStandardError * maxStandardError)));

                /**
                 * HyperLogLog(new SparseHll(indexBitLength(numberOfBuckets)):
                 * @link https://github.com/airlift/airlift/blob/5367941b6997c0d5dac3ab7263c88c4a4d002c90/stats/src/main/java/io/airlift/stats/cardinality/HyperLogLog.java#L35-L40
                 * @link https://github.com/airlift/airlift/blob/5367941b6997c0d5dac3ab7263c88c4a4d002c90/stats/src/main/java/io/airlift/stats/cardinality/SparseHll.java#L55-L61
                 */
                this.indexBitLength = (byte) indexBitLength(numberOfBuckets);
                this.entries = new int[1];

            }
            insertHash(MurmurHash3.murmurhash3_x86_128(x)); // x would be string (slice) or long => returns long
        }

        /**
         * @link https://github.com/airlift/airlift/blob/5367941b6997c0d5dac3ab7263c88c4a4d002c90/stats/src/main/java/io/airlift/stats/cardinality/SparseHll.java#L87-L118
         */
        private void insertHash(long hash) {
            int bucket = computeIndex(hash, EXTENDED_PREFIX_BITS);
            int position = searchBucket(bucket);
        }

        private static int log2Ceiling(int value) {
            return Integer.highestOneBit(value - 1) << 1;
        }

        private static int indexBitLength(int numberOfBuckets) {
            return (int) (Math.log(numberOfBuckets) / Math.log(2));
        }

        private static int computeIndex(long hash, int indexBitLength) {
            return (int) (hash >>> (Long.SIZE - indexBitLength));
        }
    }
}
