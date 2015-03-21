/**
 * 
 */
package org.numenta.nupic.util;

import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.AbstractDistance;
import net.sf.javaml.distance.DistanceMeasure;

/**
 * @author PulinTablet
 *
 */
public class HammingDistance extends AbstractDistance{

	@Override
	public double measure(Instance arg0, Instance arg1) {
		double distance=0;
		for (int i = 0; i < arg0.noAttributes(); i++) {
			distance+=Math.abs(arg0.value(i)-arg1.value(i));
		}
		return distance;
	}

}
