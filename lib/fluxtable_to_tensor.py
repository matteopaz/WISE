def fluxtable_to_tensor(self, flux): # IMPORTANT! Defines order of data
      # Len(pts) x 5 matrix
      w1 = flux["norm"]["w1"]
      w1sig = flux["norm"]["w1sig"]
      w2 = flux["norm"]["w2"]
      w2sig = flux["norm"]["w2sig"]
      dt = flux["norm"]["dt"]
      day = flux["norm"]["day"]

      w1f = flux["norm"]["w1flux"]

      std_val = (flux["norm"]["w1std"] + flux["norm"]["w2std"]) / 2

      std = np.array([std_val for _ in w1])

      # Len(pts) x 5 matrix
      # IMPORTANT! Defines order of data
      return np.stack((w1f, std, day), axis=0).T