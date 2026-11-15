"""Numerical and live dependency-graph checks for the tutorial.

Run from any directory with the notebook's dependencies installed:
    uv run --with marimo --with numpy --with matplotlib python docs/blog/validate_looped_transformer.py
"""
import importlib.util
from pathlib import Path
import unittest

import marimo as mo
import numpy as np


class TutorialChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("looped_transformer.py")
        spec = importlib.util.spec_from_file_location("looped_tutorial", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.outputs, cls.d = cls.module.app.run()

    def test_corpus_and_actual_training(self):
        d = self.d
        self.assertEqual(len(d["VOCAB"]), 11)
        self.assertEqual(d["CORPUS"], ["the cat sat on the mat"] * 3
                         + ["the cat sat on the hat", "the cat sat on the grass"])
        successes = 0
        for (seed, name), run in d["comparison_runs"].items():
            loss, _, probabilities, weights = d["comparison_loss_grad"](run["params"], name == "B")
            np.testing.assert_allclose(loss, run["losses"][-1], atol=1e-12)
            np.testing.assert_allclose(probabilities, run["p"])
            np.testing.assert_allclose(weights.sum(axis=1), 1)
            self.assertLess(run["losses"][-1], run["losses"][0])
            self.assertFalse(np.allclose(run["params"]["U"], d["comparison_init"](seed)["U"]))
            if name == "A":
                np.testing.assert_allclose(probabilities, np.repeat(probabilities[:1], 6, axis=0))
                self.assertAlmostEqual(loss, np.log(3), delta=2e-5)
                self.assertLessEqual(np.mean(probabilities.argmax(1) == d["SECOND_TARGETS"]), 1/3)
            else:
                successes += int(loss < 0.05 and np.all(probabilities.argmax(1) == d["SECOND_TARGETS"]))
        # Demonstrate capability without hiding unsuccessful seeds or demanding guaranteed optimization.
        self.assertGreaterEqual(successes, 1)
        again = d["comparison_train"](7, True)
        np.testing.assert_array_equal(again["p"], d["comparison_runs"][7, "B"]["p"])

    def test_gradients(self):
        d = self.d
        eps = 1e-6
        for allow_earlier in (False, True):
            for params in (d["comparison_init"](7), d["comparison_train"](7, allow_earlier, steps=20)["params"]):
                _, grad, _, _ = d["comparison_loss_grad"](params, allow_earlier)
                for key, array in params.items():
                    numeric = np.zeros_like(array)
                    for index in np.ndindex(array.shape):
                        old = array[index]
                        array[index] = old + eps
                        up = d["comparison_loss_grad"](params, allow_earlier)[0]
                        array[index] = old - eps
                        down = d["comparison_loss_grad"](params, allow_earlier)[0]
                        array[index] = old
                        numeric[index] = (up - down) / (2 * eps)
                    np.testing.assert_allclose(grad[key], numeric, atol=2e-7, rtol=1e-4,
                                               err_msg=f"comparison gradient {key}, attention={allow_earlier}")
        for context in (d["QUESTION"], "a dog", "the cat sat on the mat on a"):
            f = d["target"]
            table = d["E"].copy()
            analytic = d["slope_of_loss"](table, context, f)
            numeric = np.zeros_like(table)
            for index in np.ndindex(table.shape):
                up, down = table.copy(), table.copy()
                up[index] += eps
                down[index] -= eps
                numeric[index] = (d["nll"](d["first_predictor"](up, context)[2], f)
                                  - d["nll"](d["first_predictor"](down, context)[2], f)) / (2 * eps)
            np.testing.assert_allclose(analytic, numeric, atol=2e-7)
        for gain in (0, 2.5, 4.5):
            h = d["h_site"]
            pre = h @ d["W_IN"] + d["B_IN"]
            analytic = np.eye(2) + d["w_out"](gain).T @ np.diag(1 - np.tanh(pre)**2) @ d["W_IN"].T
            numeric = d["jacobian"](lambda x: d["after_mlp"](x, gain), h)
            np.testing.assert_allclose(analytic, numeric, atol=2e-8)

    def check_live_identities(self, d):
        n = len(d["words"])
        self.assertEqual(d["X"].shape, (n, 2))
        np.testing.assert_allclose(d["X"], d["E"][d["ids"]] + d["P"])
        for suffix in ("1", "2"):
            np.testing.assert_allclose(d["A"+suffix].sum(1), 1)
            np.testing.assert_allclose(np.triu(d["A"+suffix], 1), 0)
            np.testing.assert_allclose(d["AV"+suffix], d["A"+suffix] @ d["V"+suffix])
        np.testing.assert_allclose(d["Y1"], d["layer"](d["X"], d["heads"], d["gain"].value))
        np.testing.assert_allclose(d["loop_step"](d["X"], d["X"], d["heads"], d["gain"].value), d["Y1"])
        self.assertTrue(np.all(np.isfinite(d["diag_h"])))
        self.assertEqual(d["fixed_valid"], d["residual"] < 1e-8)
        actual = np.linalg.norm(d["loop_step"](d["H_star"], d["X"], d["heads"], d["gain"].value)-d["H_star"])
        self.assertAlmostEqual(actual, d["residual"], delta=1e-12)
        for k in (0, d["K_DIAG"]-1):
            # Reported refit accuracies must be held-out, not training or all-data scores.
            mids = np.array([d["run_loop"](d["embed"](c), d["heads"], d["gain"].value, k+1)[1][-1,-1]
                             for c in d["DIAG"]])
            u,b = d["train_probe"](mids[d["train_idx"]], d["diag_y"][d["train_idx"]], steps=800)
            self.assertEqual(d["acc_refit"][k], d["accuracy"](mids[d["test_idx"]],d["diag_y"][d["test_idx"]],u,b))

    def test_default_identities_and_interventions(self):
        d = self.d
        self.check_live_identities(d)
        np.testing.assert_allclose(d["measured"](np.zeros(2)), 0, atol=1e-12)
        for direction in d["directions"].values():
            push = 1e-5 * direction
            np.testing.assert_allclose(d["measured"](push), d["predicted"](push), atol=2e-8)
            once = d["measured"](push, every=False)
            predicted_once = np.array([np.linalg.matrix_power(d["J_last"],t) @ push @ d["d_place"]
                                       for t in range(1,d["T_PUSH"]+1)])
            np.testing.assert_allclose(once, predicted_once, atol=2e-8)
        np.testing.assert_array_equal(d["unit"](np.zeros(2)), np.zeros(2))

    def test_live_alternative_contexts(self):
        for context in ("on the mat sat the", "a dog", "the cat sat on the mat on a"):
            with self.subTest(context=context):
                _, d = self.module.app.run(defs={"text": context})
                self.check_live_identities(d)

    def test_live_slider_endpoints_and_zero_direction(self):
        for gain, beta, gamma, theta, phi in ((0,0,0,-180,-180), (4.5,6,2,180,180)):
            with self.subTest(gain=gain, beta=beta, gamma=gamma):
                defs = {"gain":mo.ui.slider(0,4.5,step=0.1,value=gain)}
                for suffix in ("1","2"):
                    defs.update({"beta"+suffix:mo.ui.slider(0,6,value=beta),
                                 "theta"+suffix:mo.ui.slider(-180,180,value=theta),
                                 "gamma"+suffix:mo.ui.slider(0,2,value=gamma),
                                 "phi"+suffix:mo.ui.slider(-180,180,value=phi)})
                _, d = self.module.app.run(defs=defs)
                self.check_live_identities(d)
                self.assertEqual(d["gain"].value, gain)
                self.assertEqual(d["gamma1"].value, gamma)
                if gamma == 0:
                    np.testing.assert_allclose(d["probe_u"],0,atol=1e-12)
                    np.testing.assert_allclose(d["probe_dir"],0,atol=1e-12)
                    np.testing.assert_allclose(d["r_actual"], d["r_actual"][0])
                    self.assertEqual(d["accuracy"](d["diag_h"][d["test_idx"]], d["diag_y"][d["test_idx"]],
                                                   d["probe_u"],d["probe_b"]),0.5)

    def test_failed_solver_and_divergent_run_are_labeled(self):
        opts = {}
        for i in (1, 2):
            opts.update({f"beta{i}": mo.ui.slider(0, 6, value=0),
                         f"theta{i}": mo.ui.slider(-180, 180, value=0),
                         f"gamma{i}": mo.ui.slider(0, 2, step=0.25, value=0.5),
                         f"phi{i}": mo.ui.slider(-180, 180, value=0)})
        outputs, d = self.module.app.run(defs=opts)
        self.assertFalse(d["fixed_valid"])
        self.assertGreater(d["residual"], 1e-8)
        html = " ".join(t for out in outputs if isinstance(t := getattr(out, "text", None), str))
        self.assertIn("exploratory", html)
        self.assertIn("Fixed-point intervention comparison omitted", html)
        self.assertNotIn("Locally asymptotically stable according to", html)
        # The same test with stronger aligned values exceeds the safety threshold.
        for i in (1, 2):
            opts[f"gamma{i}"] = mo.ui.slider(0, 2, value=2)
        outputs, d = self.module.app.run(defs=opts)
        html = " ".join(t for out in outputs if isinstance(t := getattr(out, "text", None), str))
        self.assertIn("numerical safety limit", html)
        self.assertNotIn("loop_states", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
