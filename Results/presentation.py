# presentation.py

from manim import *

class RLProjectPresentation(Scene):
    def construct(self):
        # Title Slide
        title = Text("Deep Q-Networks (DQN) in Reinforcement Learning").scale(1.2)
        subtitle = Text("Understanding the Math Behind Each Episode and Neuron Interactions").next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(3)
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Introduction to DQN
        intro_title = Text("Introduction to Deep Q-Networks (DQN)").to_edge(UP)
        intro_content = Tex(
            r"""
            DQN combines Q-Learning with Deep Neural Networks to handle high-dimensional state spaces.
            \\
            \textbf{Objective}: Learn the optimal action-value function $Q^*(s, a)$.
            """
        ).next_to(intro_title, DOWN)
        self.play(Write(intro_title))
        self.play(Write(intro_content))
        self.wait(4)
        self.play(FadeOut(intro_title), FadeOut(intro_content))
        
        # Neural Network Architecture
        nn_title = Text("Neural Network Architecture").to_edge(UP)
        # Simple NN Diagram
        input_layer = Circle(radius=0.3, color=BLUE).shift(LEFT*4)
        hidden_layer = VGroup(*[Circle(radius=0.2, color=GREEN) for _ in range(5)]).arrange(DOWN, buff=0.6).shift(RIGHT*1)
        output_layer = VGroup(*[Circle(radius=0.2, color=RED) for _ in range(2)]).arrange(DOWN, buff=0.6).shift(RIGHT*4)
        
        connections_input_hidden = VGroup(*[
            Line(input_layer.get_right(), neuron.get_left()) for neuron in hidden_layer
        ])
        connections_hidden_output = VGroup(*[
            Line(neuron.get_right(), out_neuron.get_left()) for neuron in hidden_layer for out_neuron in output_layer
        ])
        
        nn_diagram = VGroup(input_layer, hidden_layer, output_layer,
                            connections_input_hidden, connections_hidden_output)
        
        self.play(Write(nn_title))
        self.play(LaggedStartMap(FadeIn, nn_diagram, shift=RIGHT))
        self.wait(4)
        self.play(FadeOut(nn_title), FadeOut(nn_diagram))
        
        # Flow of an Episode
        flow_title = Text("Flow of an Episode").to_edge(UP)
        flow_content = Tex(
            r"""
            1. \textbf{State} $s_t$ \rightarrow 
            2. \textbf{Action} $a_t$ \rightarrow 
            3. \textbf{Reward} $r_t$ \rightarrow 
            4. \textbf{Next State} $s_{t+1}$
            """
        ).next_to(flow_title, DOWN)
        self.play(Write(flow_title))
        self.play(Write(flow_content))
        self.wait(4)
        self.play(FadeOut(flow_title), FadeOut(flow_content))
        
        # Neuron Activation Visualization
        activation_title = Text("Neuron Activation During an Episode").to_edge(UP)
        # Simple NN with Activation
        nn_act = NeuralNetwork(
            layer_sizes=[4, 128, 128, 2],
            neuron_radius=0.15,
            neuron_config={"color": BLUE},
            layer_config={"color": WHITE}
        ).scale(0.7)
        
        # Example activation: Highlight some neurons
        activated_neurons = VGroup(*[
            nn_act.layers[1].neurons[10],  # Random neurons in first hidden layer
            nn_act.layers[1].neurons[50],
            nn_act.layers[2].neurons[30],
            nn_act.layers[3].neurons[1]
        ])
        
        self.play(Write(activation_title))
        self.play(Create(nn_act))
        self.wait(2)
        self.play(activated_neurons.animate.set_fill(YELLOW))
        self.wait(3)
        self.play(FadeOut(activation_title), FadeOut(nn_act))
        
        # Q-value Update Mechanism
        q_update_title = Text("Q-value Update Mechanism").to_edge(UP)
        q_formula = Tex(
            r"""
            Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right)
            """
        ).next_to(q_update_title, DOWN)
        self.play(Write(q_update_title))
        self.play(Write(q_formula))
        self.wait(4)
        self.play(FadeOut(q_update_title), FadeOut(q_formula))
        
        # Backpropagation and Weight Updates
        bp_title = Text("Backpropagation and Weight Updates").to_edge(UP)
        bp_content = Tex(
            r"""
            \textbf{Loss} = \left( r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right)^2
            \\
            \text{Minimize Loss via Gradient Descent to update weights.}
            """
        ).next_to(bp_title, DOWN)
        self.play(Write(bp_title))
        self.play(Write(bp_content))
        self.wait(4)
        self.play(FadeOut(bp_title), FadeOut(bp_content))
        
        # Conclusion
        conclusion_title = Text("Conclusion").to_edge(UP)
        conclusion_content = Tex(
            r"""
            DQN leverages deep neural networks to approximate the optimal Q-function.
            \\
            Through episodes, neurons adjust weights to maximize expected rewards.
            """
        ).next_to(conclusion_title, DOWN)
        self.play(Write(conclusion_title))
        self.play(Write(conclusion_content))
        self.wait(4)
        self.play(FadeOut(conclusion_title), FadeOut(conclusion_content))
        
        # End Slide
        end_title = Text("Thank You!").scale(1.5)
        self.play(Write(end_title))
        self.wait(2)
        self.play(FadeOut(end_title))

# NeuralNetwork class to create a simple NN diagram
class NeuralNetwork(VGroup):
    def __init__(self, layer_sizes, neuron_radius=0.15, neuron_config={}, layer_config={}, **kwargs):
        super().__init__(**kwargs)
        self.layers = VGroup()
        for idx, size in enumerate(layer_sizes):
            layer = VGroup(*[
                Circle(radius=neuron_radius, **neuron_config).shift(DOWN * i * (neuron_radius * 3))
                for i in range(size)
            ])
            layer.arrange(DOWN, buff=0.6)
            layer.shift(RIGHT * idx * 2)
            self.layers.add(layer)
        self.add(self.layers)
        # Connect neurons
        connections = VGroup()
        for l in range(len(layer_sizes) - 1):
            for neuron in self.layers[l]:
                for next_neuron in self.layers[l + 1]:
                    connection = Line(neuron.get_right(), next_neuron.get_left(), stroke_width=1)
                    connections.add(connection)
        self.add(connections)
