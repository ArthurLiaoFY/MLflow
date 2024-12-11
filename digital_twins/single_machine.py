# %%
# import pika
import salabim as sim


class SNGenerator(sim.Component):
    def process(self):
        while True:
            while head_buffer.available_quantity() <= 0:
                self.standby()
            # channel.basic_publish(
            #     exchange="arthur_direct_logs",
            #     routing_key=severity,
            #     body="SN generated",
            # )
            self.request(conveyor1)
            self.hold(2)
            self.release()
            # channel.basic_publish(
            #     exchange="arthur_direct_logs",
            #     routing_key=severity,
            #     body="SN sended to head buffer",
            # )
            SN().enter(head_buffer)


class SN(sim.Component):
    def process(self):
        while True:
            self.passivate()

class SNSink(sim.Component):
    def process(self):
        while True:
            while tail_buffer.available_quantity() == tail_buffer.capacity.value:
                self.standby()
            product = self.from_store(tail_buffer)
            # channel.basic_publish(
            #     exchange="arthur_direct_logs",
            #     routing_key=severity,
            #     body="SN leaving tail buffer",
            # )
            self.request(conveyor2)
            self.hold(4)
            self.release()
            product.passivate()
            # channel.basic_publish(
            #     exchange="arthur_direct_logs",
            #     routing_key=severity,
            #     body="SN process done",
            # )
            env.total_prod_amount += 1


class Machine(sim.Component):
    def setup(self):
        self.machine_status = {
            status_code: sim.State(name=cn_name, value=False)
            for status_code, cn_name in zip(
                range(-1, 5),
                [
                    "Normal",
                    "Fail",
                    "WaitingForReceive",
                    "WaitingForDeliver",
                    "Down",
                ],
            )
        }

    def switch_to_status(self, status: int):
        # channel.basic_publish(
        #     exchange="arthur_direct_logs",
        #     routing_key=severity,
        #     body=f"machine status switch to {status}",
        # )
        for status_code in self.machine_status:
            if status_code == 4:
                self.machine_status[status_code].set(value=True)

            else:
                self.machine_status[status_code].set(value=False)

    def process(self):
        while True:
            while len(head_buffer) == 0:
                self.switch_to_status(status=4)
                self.standby()
            # channel.basic_publish(
            #     exchange="arthur_direct_logs",
            #     routing_key=severity,
            #     body="machine receive sn from head buffer",
            # )
            product = self.from_store(head_buffer)

            if self.ispassive():
                self.activate()
            self.switch_to_status(status=0)

            self.hold(sim.Exponential(5))

            while tail_buffer.available_quantity() <= 0:
                self.switch_to_status(status=5)
                self.standby()

            if self.ispassive():
                self.activate()
            self.switch_to_status(status=0)
            # channel.basic_publish(
            #     exchange="arthur_direct_logs",
            #     routing_key=severity,
            #     body="machine pass sn to tail buffer",
            # )
            self.to_store(tail_buffer, product)


# connection = pika.BlockingConnection(
#     pika.ConnectionParameters(host="localhost", port="5672")
# )
# channel = connection.channel()
# channel.exchange_declare(exchange="single_machine_simulator", exchange_type="direct")

severity = "0"
animate = False
run_till = 50
seed = 1122

env = sim.Environment(trace=True, random_seed=seed)
env.total_prod_amount = 0

sn_generator = SNGenerator(name="ProductLauncher")
conveyor1 = sim.Resource("Conveyer")
head_buffer = sim.Store(name="Buffer", capacity=8)
machine = Machine(name="EquipmentA")
tail_buffer = sim.Store(name="Buffer", capacity=8)
conveyor2 = sim.Resource("Conveyer")
sn_sink = SNSink(name="ProductSink")
env.run(run_till)
# %%
sn_generator.status.print_histogram(values=True)
machine.status.print_histogram(values=True)
sn_sink.status.print_histogram(values=True)


head_buffer.print_statistics()
tail_buffer.print_statistics()
# %%
