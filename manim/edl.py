import numpy as np
from manimlib import *

import os


class Edl(Scene):

    def construct(self):
        # title -> to top
        title = Text(
            "Evidential Neural Network vs. Traditional Neural Network",
            font="Consolas",
            font_size=30,
            t2c={"Evidential Neural Network": BLUE,
                 "Traditional Neural Network": ORANGE}).scale(0.8)

        self.play(FadeInFromPoint(title, UP * 3))
        self.wait()
        self.play(
            title.animate.to_edge(UP, buff = 0.1)
        )

        self.wait()
        # traditional network work text
        tnn_title = Text(
            "How does Traditional Neural Networks work?",
            font="Consolas",
            font_size=26,
            t2c={"Traditional Neural Networks": ORANGE}
        ).to_corner(UL)
        self.play(FadeInFromPoint(tnn_title, UL * 4))
        self.wait()

        # first part - tnn process
        self.show_neural_net_process()

        self.play(FadeOutToPoint(tnn_title, UR * 4))
        tnn_problem_title = Text(
            "What are the problems in traditional Neural Networks?",
            font="Consolas",
            font_size=25,
            t2c={"traditional Neural Networks": ORANGE},
        ).to_corner(UL)
        self.play(FadeInFromPoint(tnn_problem_title, UL * 4))
        self.wait()
        # second part - problem in tnn
        self.show_problems_in_tnn(tnn_problem_title)

        self.play(FadeOutToPoint(tnn_problem_title, UR * 4))
        how_enn_work_text = Text(
            "How the Evidential Neural Networks work?",
            font="Consolas",
            font_size=25,
            t2c={"Evidential Neural Networks": BLUE}
        ).to_corner(UL)
        self.play(FadeInFromPoint(how_enn_work_text, UL * 4))
        self.wait()

        edl_network = self.show_enn_process()

        self.play(FadeOutToPoint(how_enn_work_text, UR * 4))

        comparation_text = Text(
            "Comparation",
            font="Consolas",
            font_size = 30,
        ).to_corner(UL)
        self.play(FadeInFromPoint(comparation_text, UL * 4))
        self.wait()

        self.show_comparation(edl_network)

        self.play(
            FadeOutToPoint(comparation_text, UL * 4),
            FadeOutToPoint(title, UP * 4)
        )

        self.play(
            GrowFromCenter(Text(
                "Thanks for watching",
                gradient=(BLUE, GREEN),
                font_size=50,
                t2f={'Thanks for watching': 'Forte'}
            ).move_to(ORIGIN)
        )
        )


    def show_comparation(self, edl_network):

        one_image, _ = self.load_image("./images/digit.jpg")
        yoda_image, _ = self.load_image("./images/yoda.jpg")

        self.play(
            FadeIn(one_image, RIGHT)
        )
        self.wait(2)
        tnn_network = edl_network[0].copy().next_to(one_image, UR + RIGHT).scale(0.5)
        edl_network.scale(0.6).next_to(one_image, DR + RIGHT)

        img2tnn_arrow = self.create_arrow(one_image, tnn_network).scale(0.5)
        img2edl_arrow = self.create_arrow(one_image, edl_network).scale(0.5)

        self.play(
            AnimationGroup(
                FadeIn(img2edl_arrow),
                FadeIn(img2tnn_arrow)
            )
        )
        self.play(
            AnimationGroup(
                FadeIn(tnn_network, UP),
                FadeIn(edl_network, DOWN)
            )
        )
        self.wait(2)

        digit_prediction = ImageMobject(
            "./images/digit_prediction.png",
            height=4
        ).to_edge(RIGHT)
        yoda_prediction = ImageMobject(
            "./images/yoda_prediction.png",
            height=4
        ).to_edge(RIGHT)

        edl2prediciton_arrow = self.create_arrow(edl_network, digit_prediction).scale(0.5)
        tnn2prediciton_arrow = self.create_arrow(tnn_network, digit_prediction).scale(0.7)
        self.play(
            AnimationGroup(
                FadeIn(edl2prediciton_arrow, RIGHT),
                FadeIn(tnn2prediciton_arrow, RIGHT),
                FadeInFromPoint(digit_prediction, LEFT * 3),
            )
        )

        self.wait(5)

        self.play(
            FadeOutToPoint(digit_prediction, RIGHT * 3),
            FadeTransform(one_image, yoda_image),
        )
        self.wait(2)
        self.play(
            FadeIn(yoda_prediction)
        )
        self.wait(5)

        self.remove(
            yoda_image,
            yoda_prediction,
            edl_network,
            tnn_network,
            img2tnn_arrow,
            img2edl_arrow,
            tnn2prediciton_arrow,
            edl2prediciton_arrow
        )

    def show_enn_process(self):

        network = Rectangle(3, 2, color=BLUE).set_opacity(1).to_edge(LEFT)
        network.add(Text("Training Net", font="Consolas", font_size=25).move_to(network.get_center()))

        softmax_vector= np.array([
            [0.042],
            [0.315],
            [0.002],
            [0.007],
            [0.057],
            [0.211],
            [0.015],
            [0.315],
            [0.005],
            [0.025]], np.float64)

        evidence_vector = np.array([
            [3.00],
            [5.00],
            [0.10],
            [1.20],
            [3.30],
            [4.60],
            [2.00],
            [5.00],
            [1.00],
            [2.50]], np.float64)


        alpha = evidence_vector + 1
        S = alpha.sum()
        belief = evidence_vector / S
        p_hat = alpha / S
        K = 10
        uncertainty = round(K / S, 3)

        softmax_output = self.create_vector(softmax_vector, network, text=r"y_p")
        net2softmax_arrow = self.create_arrow(network, softmax_output[0], text=r"softmax")

        self.play(
            FadeIn(network.to_edge(LEFT), RIGHT),
        )
        self.wait()
        self.play(
            FadeIn(net2softmax_arrow, RIGHT),
            FadeIn(softmax_output, RIGHT)
        )

        self.wait(2)

        evidence = self.create_vector(evidence_vector, network, text=r"y_p/Evidence")
        net2evidence_arrow = self.create_arrow(network, evidence[0], text=r"ReLu")

        self.play(
            AnimationGroup(
                ReplacementTransform(softmax_output, evidence),
                ReplacementTransform(net2softmax_arrow, net2evidence_arrow)
            )
        )

        moved_evidence = evidence.copy().to_edge(LEFT)
        alpha = self.create_vector(evidence_vector + 1, moved_evidence[0], r"\alpha", interval=4)
        evidence2alpha_arrow = self.create_arrow(moved_evidence[0], alpha[0], text=" + 1")
        enn_probability = self.create_vector(p_hat, alpha[0], text=r"y_{\hat p}", interval = 6)
        alpha2proba_arrow = self.create_arrow(alpha[0], enn_probability[0], text=r"\hat p = \frac {\alpha} {S}")
        alpha2proba_arrow.add(Tex(fr"u = \frac K S = {uncertainty}").next_to(alpha2proba_arrow, DOWN).scale(0.5))

        s_calculate_text = Tex(
            fr"S = \sum_i^K (\bold \alpha_i - 1) = \sum_i^K \bold e_i = {S}",
            font_size=25
        ).next_to(alpha, DOWN * 2, aligned_edge=LEFT)

        uncertainty_calculate_text = Tex(
            fr"u = \frac K S = {uncertainty}",
            font_size=25
        ).next_to(s_calculate_text, DOWN, aligned_edge=LEFT)

        enn_loss = Tex(
            r"L(\Theta) = \sum_{i=1}^{N} L^p_i(\Theta) + \lambda_t L^e_i(\Theta)",
            font_size=30
        ).next_to(enn_probability[0], RIGHT * 2 + UP , aligned_edge=UP)

        prediction_loss = Tex(
            r"L^p_i(\Theta) = \sum_{j=1}^{k}(y_{ij} - \hat p_{ij})^2 + \frac {\hat p_{ij}(1 - \hat p_{ij})}{S_i + 1}",
            font_size=30
        ).next_to(enn_loss, DOWN * 2, aligned_edge=LEFT)

        evidence_loss = Tex(
            r"L^e_i(\Theta) = log(\frac {\Gamma(\sum_{k=1}^{K}\widetilde \alpha_{ik})}"
            r"{\Gamma(K)\prod_{k=1}^{K}\Gamma(\widetilde \alpha_{ik})}) + "
            r"\sum_{k=1}^{K}(\widetilde \alpha_{ik} - 1)[\psi(\widetilde "
            r"\alpha_{ik}) - \psi(\sum_{j=1}^{K} \widetilde \alpha_{ij})]",
            font_size=30
        ).next_to(prediction_loss, DOWN * 2, aligned_edge=LEFT)


        self.play(
            AnimationGroup(
                FadeOut(network),
                FadeOut(net2evidence_arrow),
                FadeTransform(evidence, moved_evidence)
            )
        )
        self.wait(2)

        enn_process = VGroup(*[
            moved_evidence,
            evidence2alpha_arrow,
            s_calculate_text,
            uncertainty_calculate_text,
            alpha,
            alpha2proba_arrow,
            enn_probability,
            enn_loss,
            prediction_loss,
            evidence_loss
        ])
        evidence_part = Rectangle(3, 2, color = ORANGE).set_opacity(1).next_to(network, RIGHT * 4)
        evidence_part.add(Text("evidence adjustement part",
                               font_size=30,
                               font="Consolas").move_to(evidence_part.get_center()).scale(0.5))

        self.play(
            AnimationGroup(
                FadeIn(evidence2alpha_arrow, RIGHT),
                FadeIn(alpha, RIGHT)
            )
        )

        self.wait(2)
        self.play(
            AnimationGroup(
                Write(s_calculate_text),
                Write(uncertainty_calculate_text)
            )
        )
        self.wait(2)

        self.play(
            AnimationGroup(
                FadeIn(alpha2proba_arrow, RIGHT),
                FadeIn(enn_probability, RIGHT)
            )
        )
        self.wait(2)

        self.play(
            AnimationGroup(
                FadeIn(enn_loss, RIGHT),
            )
        )
        self.wait()
        self.play(
            AnimationGroup(
                FadeIn(prediction_loss),
                FadeIn(evidence_loss)
            )
        )
        self.wait(2)


        intergrate_part = VGroup(
            network,
            net2evidence_arrow,
            evidence_part
        )

        self.play(
            FadeIn(network, RIGHT),
            FadeIn(net2evidence_arrow, RIGHT),
            FadeTransform(enn_process, evidence_part)
        )
        self.wait(2)

        intergrate_part.background_rectangle = SurroundingRectangle(intergrate_part, color=BLACK).set_opacity(0.6)
        intergrate_part.text = Text("EDL network", font_size = 25, font="Consolas").scale(0.8).next_to(intergrate_part, UP)
        self.play(
            FadeIn(intergrate_part.background_rectangle),
            FadeIn(intergrate_part.text)
        )
        self.wait(2)
        intergrate_part = VGroup(
            *intergrate_part,
            intergrate_part.background_rectangle,
            intergrate_part.text
        )

        self.play(
            intergrate_part.animate.move_to(ORIGIN)
        )
        self.wait()

        example_img, img_lable = self.load_image("./images/mnistex.jpg", text="train example 1")

        img2edlnet_arrow = self.create_arrow(example_img, intergrate_part, text="Input")

        enn_probability.next_to(intergrate_part, RIGHT * 4)
        edlnet2output_arrow = self.create_arrow(intergrate_part, enn_probability, text="output")
        edlnet2output_arrow.add(Text(fr"u={uncertainty}", font_size=25, font="Consolas")
                                .next_to(edlnet2output_arrow, DOWN).scale(0.8))
        final_group = Group(
            example_img,
            img_lable,
            img2edlnet_arrow,
            edlnet2output_arrow,
            enn_probability
        )
        self.play(
            AnimationGroup(
            FadeIn(example_img, RIGHT),
            FadeIn(img_lable, RIGHT),
            FadeIn(img2edlnet_arrow, RIGHT)
            )
        )
        self.wait(2)

        self.play(
            AnimationGroup(
                FadeIn(edlnet2output_arrow, RIGHT),
                FadeIn(enn_probability, RIGHT)
            )
        )
        self.wait(2)

        self.remove(
            intergrate_part,
            *final_group
        )

        return intergrate_part

    def show_problems_in_tnn(self, tnn_problem_title):

        # get minist image and its size label
        minist_img, _ = self.load_image("./images/mnistex.jpg")

        trained_net = Rectangle(3, 2, color=GREEN).set_opacity(1).next_to(minist_img, RIGHT*4)
        trained_net.add(Text("Trained Net", font= "Consolas").scale(0.4).move_to(trained_net.get_center()))

        img2net_arrow = self.create_arrow(minist_img, trained_net, text="Input")

        overconfident_output = self.create_vector(np.array([
            [0.25],
            [0.3],
            [0.25],
            [0.2],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0]
        ], np.float64), trained_net, text="y_p")
        overconfident_output[0].add(SurroundingRectangle(overconfident_output[0].get_rows()[1], color=GREEN))
        net2output_arrow = self.create_arrow(trained_net, overconfident_output[0], text="softmax")


        self.play(
            FadeIn(minist_img, RIGHT)
        )
        self.wait()
        self.play(
            FadeIn(img2net_arrow, RIGHT)
        )
        self.play(
            FadeIn(trained_net, RIGHT)
        )
        self.play(
            FadeIn(net2output_arrow, RIGHT)
        )
        self.wait()
        self.play(
            FadeIn(overconfident_output, RIGHT)
        )

        overconfident_prediction = Text(
            "This Image is 1",
            font_size=20,
            t2c={"1": GREEN}
        ).next_to(overconfident_output[0], RIGHT * 3 + UP)
        overconfident_judge = Text("OVERCONFIDENT X",
                                   font_size=20,
                                   t2c={"X": RED}). \
            move_to(tnn_problem_title.get_left() + np.array((0, -0.5, 0)), LEFT)


        overconfident_process = [
            minist_img,
            img2net_arrow,
            trained_net,
            net2output_arrow,
            overconfident_output,
            overconfident_prediction,
            overconfident_judge
        ]
        self.play(FadeIn(overconfident_prediction))
        self.wait(2)
        self.play(FadeTransform(overconfident_prediction, overconfident_judge))
        self.wait(2)

        # self.remove(
        #     *over_process,
        # )

        dog_image, _ = self.load_image("./images/dog.png", height=1.5, image_scale=0.7)


        dog_output = self.create_vector(np.array([
            [0.25],
            [0.30],
            [0.40],
            [0.01],
            [0.02],
            [0.02],
            [0.0],
            [0.0],
            [0.0],
            [0.0]
        ], np.float64), trained_net, "y_p")
        dog_output[0].add(SurroundingRectangle(dog_output[0].get_rows()[2], color=GREEN))
        # net2output_arrow = self.create_arrow(trained_net, dog_output[0], text="softmax")

        self.play(
            FadeOut(overconfident_output)
        )
        self.wait()
        self.play(
            FadeTransform(minist_img, dog_image)
        )
        self.wait()
        self.play(
            FadeIn(dog_output, RIGHT)
        )
        self.wait()

        dog_problem_prediction = Text("This Image is 2",
                                      font_size=20,
                                      t2c={"2": GREEN}).next_to(dog_output[0], RIGHT + UP)
        # self.play(Write(dog_problem_pre))
        dontno_judge = Text("Never say I don't know X",
                            font_size=20,
                            t2c={"I don't know": RED_D,
                                 "X": RED}). \
            move_to(overconfident_judge.get_left() + np.array((0, -0.5, 0)), LEFT)

        dog_process = [
            dog_image,
            dog_output,
            dog_problem_prediction,
            dontno_judge,
        ]
        self.play(FadeIn(dog_problem_prediction, RIGHT))
        self.wait(2)
        self.play(FadeTransform(dog_problem_prediction, dontno_judge))
        self.wait(2)

        self.remove(
            *overconfident_process,
            *dog_process
        )

        self.wait(2)

    def show_neural_net_process(self):

        # show minist example
        minist_img, size_label = self.load_image("./images/mnistex.jpg")

        self.play(
            AnimationGroup(
                FadeIn(minist_img, RIGHT),
                FadeIn(size_label, RIGHT)
            )
        )

        self.wait(2)
        # cnn network detail and forward propagation,
        # show_net_and_path function can create a simple LeNet type cnn graph and show the forward propagation path
        network = self.show_net_and_path(minist_img)

        # get a output vector
        output = self.create_vector(np.array([
            [0.30],
            [0.20],
            [0.20],
            [0.20],
            [0.03],
            [0.07],
            [0.02],
            [0.08],
            [0.01],
            [0.09]], np.float64), network[0], "y_p")
        # get the network to output arrow
        net2output_arrow = self.create_arrow(network[0], output[0], text="softmax")
        self.play(
            AnimationGroup(
                FadeIn(net2output_arrow, RIGHT),
                FadeIn(output, RIGHT)
            )
        )
        self.wait(2)
        # create a simple rectangle to represent the cnn network
        simple_network = Rectangle(3, 2, color=BLUE).scale(0.8).set_opacity(1).next_to(minist_img, RIGHT * 4)
        simple_network.add(Text("Traditional CNN", font="Consolas", font_size=25).scale(0.8).move_to(simple_network.get_center()))
        img2net_arrow = self.create_arrow(minist_img, simple_network, text="Input").scale(0.8)

        # create a simple architecture to represent the tnn process
        new_output = output.copy().next_to(simple_network, RIGHT * 4)
        new_net2output_arrow = self.create_arrow(simple_network, new_output, text="softmax").scale(0.8)

        # create a ground truth vector
        yt = self.create_vector(np.array([
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
            [0],
            [0],
            [0]]), new_output[0], "y_t")

        # create a text to show the tnn loss function
        tnn_loss = Tex(r"L(\Theta) = -log \ \sigma(f_y(x, \Theta))",
                       font_size=30).move_to(yt.get_right() + np.array((2, 0, 0)))

        # show the arrow from network to output, and the network output vector
        # self.play(
        #     AnimationGroup(
        #         FadeIn(net2output_arrow, RIGHT),
        #         FadeIn(output, RIGHT)
        #     )
        # )
        # remove the network detail. use a simple rectangle to represent the network

        # create a old component group to store the component being showed on the screen
        # create a new component group to store the component that are more simple
        old_group = [
            network,
            net2output_arrow,
            output
        ]

        new_group = [
            simple_network,
            new_net2output_arrow,
            new_output
        ]

        # use AnimationGroup to store the Animation, which include:
        # show the arrow from image to simple network,
        # transform the old group component to the new group component which are more simple and intuitive
        # use AnimationGroup can play the animation simultaneously
        self.play(
            AnimationGroup(
                FadeIn(img2net_arrow, RIGHT),
                *[FadeTransform(old, new) for old, new in list(zip(old_group, new_group))],
            ),
            run_time=2
        )
        self.wait(2)
        # show the ground truth vector and the loss function text

        self.play(
            FadeIn(yt),
        )
        self.wait(2)
        self.play(
            FadeInFromPoint(tnn_loss, RIGHT * 3)
        )
        self.wait(2)
        # remove all the component that show in this function
        self.remove(
            minist_img,
            size_label,
            img2net_arrow,
            *new_group,
            yt,
            tnn_loss,
        )

    def create_vector(self, matrix, aligned, text, matrix_scaler=0.5, interval=2, font_size=30):

        vector = DecimalMatrix(matrix).next_to(aligned, RIGHT * interval).scale(matrix_scaler)
        text = Tex(
            r"" + text,
            font_size=font_size
        ).next_to(vector, UP)
        group = VGroup(
            vector,
            text
        )
        return group

    def create_arrow(self, left, right, has_text=True, text="", arrow_scaler=1.2, font_size=25):
        arrow = Arrow(left.get_right(), right.get_left()).scale(arrow_scaler)
        if has_text:
            text = Tex(
                r"" + text,
                font_size=font_size
            ).next_to(arrow, UP)
            group = VGroup(
                arrow,
                text
            )
            return group
        else:
            return arrow

    def load_image(self, path, toedge=LEFT, height=2.0, image_scale=0.7, text="32X32\nINPUT"):

        img = ImageMobject(
            path,
            height=height
        ).scale(image_scale).to_edge(toedge)

        img_text = Text(text).next_to(img, DOWN).scale(0.3)

        return img, img_text

    def show_net_and_path(self, img_input):

        # first conv layer
        cov1_layer = self.create_layer_group(size=0.8, layer_name="conv1") \
            .move_to(img_input.get_right() + np.array((1.5, 0, 0)))

        # first sampling layer
        samp1_layer = self.create_layer_group(0.6, 6, size_text="6X14X14", layer_name="samp1") \
            .move_to(cov1_layer.get_right() + np.array((0.3, 0, 0)))

        # second conv layer
        cov2_layer = self.create_layer_group(0.5, 16, size_text="16X10X10", layer_name="conv2", buff=0.15) \
            .move_to(samp1_layer.get_right() + np.array((0.2, 0, 0)))

        # second sampling layer
        samp2_layer = self.create_layer_group(0.4, 16, size_text="16X5X5", layer_name="samp2", buff=0.15) \
            .move_to(cov2_layer.get_right())

        # full connection layer
        f1 = self.create_full_connection(size_text="120X1X1") \
            .move_to(samp2_layer.get_right() + np.array((0.5, 0, 0)))
        f2 = self.create_full_connection(height=3, size_text="84X1X1", layer_name="F2") \
            .move_to(f1[0].get_right() + np.array((0.5, 0, 0)))
        f3 = self.create_full_connection(height=2, size_text="10X1X1", layer_name="OUTPUT") \
            .move_to(f2[0].get_right() + np.array((0.5, 0, 0)))

        # network
        network = VGroup(
            cov1_layer,
            samp1_layer,
            cov2_layer,
            samp2_layer,
            f1,
            f2,
            f3
        )
        # forward path
        forward_path = VGroup(
            self.connect_net_part(img_input, cov1_layer[0][0], line_text="cov1"),
            self.connect_net_part(cov1_layer[0][0], samp1_layer[0][0], line_text="samp1"),
            self.connect_net_part(samp1_layer[0][0], cov2_layer[0][0], line_text="cov1", part_size=0.2),
            self.connect_net_part(cov2_layer[0][0], samp2_layer[0][0], line_text="samp2", part_size=0.1),
            Line(samp2_layer[0][0].get_corner(DR), f1[0].get_corner(DL)).set_color(RED),
            Line(f1[0].get_corner(DR), f2[0].get_corner(DL)).set_color(RED),
            Line(f2[0].get_corner(DR), f3[0].get_corner(DL)).set_color(RED),
            Line(samp2_layer[0][-1].get_corner(UR), f1[0].get_corner(UL)).set_color(RED),
            Line(f1[0].get_corner(UR), f2[0].get_corner(UL)).set_color(RED),
            Line(f2[0].get_corner(UR), f3[0].get_corner(UL)).set_color(RED),
        )

        self.play(GrowFromCenter(network))
        self.play(Write(forward_path), run_time = 8)

        return VGroup(network, forward_path)

    def connect_net_part(self, pre, next, part_size=0.3, line_color=RED, line_text="cov"):
        previous_part = Square(side_length=part_size, color=RED).move_to(pre.get_corner(UL) + np.array((0.2, -0.2, 0)))
        line = Line(previous_part.get_center(), next.get_corner(UL)).set_color(line_color)
        line_text = Text(f"{line_text}").move_to(line.get_center() + np.array((0.3, 0.1, 0))).scale(0.3).set_color(
            line_color)
        line_part = VGroup(
            line,
            line_text
        )

        part = VGroup(
            previous_part,
            line_part
        )
        return part

    def create_full_connection(self, width=0.2, height=4, size_text="120X1X1", layer_name="F1"):

        rec_ = Rectangle(width=width, height=height, color=GREY).set_opacity(1)
        flayer = VGroup(
            rec_,
            Text(f"{size_text}").scale(0.3).next_to(rec_, DOWN),
            Text(f"{layer_name}").scale(0.3).next_to(rec_, DOWN * 2),
        )
        return flayer

    def create_layer_group(self, size=1.0, num=6, size_text="6X28X28", layer_name="convolution", buff=0.3):

        layers = [Square(side_length=size, color=GREY).set_opacity(1)]
        for i in range(1, num):
            cur = layers[i - 1].copy().set_color(GREY_B).move_to(
                layers[i - 1].get_center() + np.array((buff, buff, 0)))
            layers.append(cur)
        layer_bloc = VGroup(
            *layers
        )
        layer = VGroup(
            layer_bloc,
            Text(f"{size_text}").scale(0.3).next_to(layers[0], DOWN),
            Text(f"{layer_name}").scale(0.3).next_to(layers[0], DOWN * 2),
        )
        return layer


if __name__ == '__main__':
    os.system("manimgl edl.py Edl")
