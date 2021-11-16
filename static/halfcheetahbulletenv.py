import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0] + 1.25
        pitch = next_obs[:, 7]
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .8) \
                    * (np.abs(pitch) < 1.0)
        done = ~not_done
        done = np.zeros_like(done)
        done = done[:,None]
        return done

