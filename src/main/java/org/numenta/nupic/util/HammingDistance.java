/**
 * 
 */
package org.numenta.nupic.util;

import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.AbstractDistance;

/**
 * @author PulinTablet
 *
 */
public class HammingDistance extends AbstractDistance{

	@Override
	public double measure(Instance arg0, Instance arg1) {
		double distance=0;
		for (int i = 0; i < arg0.noAttributes(); i++) {
			distance+=Math.abs(arg1.value(i)-arg0.value(i));
		}
		return distance;
	}

}
