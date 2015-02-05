/*
 * Hivemall: Hive scalable Machine Learning Library
 *
 * Copyright (C) 2013-2014
 *   National Institute of Advanced Industrial Science and Technology (AIST)
 *   Registration Number: H25PRO-1520
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */
package hivemall.mix.store;

import javax.annotation.Nonnegative;
import javax.annotation.concurrent.GuardedBy;

public final class PartialArgminKLD extends PartialResult {

    @GuardedBy("lock()")
    private double sum_mean_div_covar;
    @GuardedBy("lock()")
    private float sum_inv_covar;

    public PartialArgminKLD() {
        super();
        this.sum_mean_div_covar = 0.f;
        this.sum_inv_covar = 0.f;
    }

    @Override
    public float getCovariance(float scale) {
        return 1.f / (sum_inv_covar * scale);
    }

    @Override
    public void add(float localWeight, float covar, short clock, int deltaUpdates, float scale) {
        assert (deltaUpdates > 0) : deltaUpdates;
        addWeight(localWeight, covar, scale);
        incrClock(clock);
    }

    private void addWeight(float localWeight, float covar, float scale) {
        this.sum_mean_div_covar += (localWeight / covar) / scale;
        this.sum_inv_covar += (1.f / covar) / scale;
    }

    @Override
    public void subtract(float localWeight, float covar, @Nonnegative int deltaUpdates, float scale) {
        this.sum_mean_div_covar -= (localWeight / covar) / scale;
        this.sum_inv_covar -= (1.f / covar) / scale;
    }

    @Override
    public float getWeight(float scale) {
        return (float) (sum_mean_div_covar / sum_inv_covar);
    }

}
