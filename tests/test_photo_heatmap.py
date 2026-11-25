"""Three small geometry checks; no photos or GPU needed."""
import unittest
import numpy as np
from scripts.photo_heatmap import stitch_cells,cell_to_pixels,cell_edges,reduce_cells


class PhotoHeatmapTests(unittest.TestCase):
    def test_aligned_tiles_and_photo_edges(self):
        maps=np.stack([np.ones((2,8,8)),np.full((2,8,8),3.)])
        actual=stitch_cells((256,512),[(0,0),(0,256)],maps)
        np.testing.assert_array_equal(actual[:,:,:8],1)
        np.testing.assert_array_equal(actual[:,:,8:],3)
        with self.assertRaisesRegex(ValueError,'uncovered'):
            stitch_cells((256,512),[(0,0)],maps[:1])

    def test_overlap_last_pixel_averaging_then_partial_bins(self):
        # Origin14 is deliberately NOT a multiple of32. Two views overlap at
        # pixels14..255. The overlap is averaged before the32px photo bins.
        maps=np.stack([np.zeros((1,8,8)),np.full((1,8,8),2.)])
        actual=stitch_cells((270,256),[(0,0),(14,0)],maps)[0]
        expected=np.r_[18/32,np.ones(7),2.]
        np.testing.assert_allclose(actual,np.broadcast_to(expected[:,None],(9,8)))
        self.assertEqual(cell_edges(270).tolist(),[0,32,64,96,128,160,192,224,256,270])

    def test_cell_pixel_mapping_and_real_partial_edge_area(self):
        values=np.arange(6).reshape(2,3)
        pixels=cell_to_pixels(values,(35,70))
        self.assertEqual(pixels.shape,(35,70))
        self.assertEqual(pixels[31,31],0);self.assertEqual(pixels[32,32],4)
        self.assertEqual(pixels[-1,-1],5)
        np.testing.assert_array_equal(reduce_cells(pixels),values)
        change=np.zeros((35,70),bool);change[-1,-1]=True
        truth=reduce_cells(change)>0
        np.testing.assert_array_equal(np.argwhere(truth),[[1,2]])
        self.assertAlmostEqual(reduce_cells(change)[-1,-1],1/(3*6))


if __name__=='__main__':unittest.main()
